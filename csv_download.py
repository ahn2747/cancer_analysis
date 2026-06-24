from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
import os
import time
import shutil

# =====================================================================
# 디렉토리 설정 (분석 파이프라인과 동일한 구조 유지)
# =====================================================================
try:
    super_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    super_dir = os.path.abspath(".")

# 다운로드를 임시로 받을 폴더 (충돌 방지)
temp_download_dir = os.path.join(super_dir, "temp_download")
os.makedirs(temp_download_dir, exist_ok=True)

def setup_driver():
    """
    Selenium WebDriver를 설정합니다.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') # 처음엔 눈으로 확인하고, 잘 되면 주석 해제하세요!
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    # 불필요한 크롬 시스템 에러 로그(DEPRECATED_ENDPOINT 등) 숨기기
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # 다운로드 폴더 자동 지정 옵션
    prefs = {
        "download.default_directory": temp_download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    return driver

def clear_temp_folder():
    """새 파일을 다운받기 전 임시 폴더를 깨끗하게 비웁니다."""
    for filename in os.listdir(temp_download_dir):
        file_path = os.path.join(temp_download_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            pass

def wait_for_download(timeout=30):
    """파일 다운로드가 완료될 때까지 대기합니다 (.crdownload 확장자가 없어질 때까지)"""
    seconds = 0
    while seconds < timeout:
        files = os.listdir(temp_download_dir)
        if files and all(not f.endswith('.crdownload') for f in files):
            # 다운로드 완료된 첫 번째 파일명 반환
            return files[0]
        time.sleep(1)
        seconds += 1
    return None

def download_gene_data(driver, gene, cancer_type):
    """
    특정 유전자와 암종에 대한 p-value를 추출하고 CSV 파일을 다운로드합니다.
    """
    # 대기 시간 10초로 원복
    wait = WebDriverWait(driver, 10) 
    
    target_dir = os.path.join(super_dir, f"input_{cancer_type}_csv")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # 1. 메인 검색 페이지 접속
        driver.get("http://www.oncolnc.org/")
        
        # 2. 유전자 검색 칸 찾기 및 입력
        search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
        search_box.clear()
        search_box.send_keys(gene)
        search_box.send_keys(Keys.RETURN)
        
        # 3. 암종(cancer_type) 버튼 찾기
        xpath_btn = f"//form[input[@name='cancer' and @value='{cancer_type}']]//button[@type='submit']"
        try:
            btn = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_btn)))
        except TimeoutException:
            print(f"  [안내] {gene} - {cancer_type}: 해당 유전자가 DB에 없거나 검색 속도가 너무 느립니다.")
            return None
            
        # 새 탭(target="_blank") 열림 방지
        form_element = btn.find_element(By.XPATH, "..")
        driver.execute_script("arguments[0].removeAttribute('target');", form_element)
        btn.click()
        
        # 4. 결과 설정(Kaplan) 페이지: Percentile 입력
        lower_input = wait.until(EC.presence_of_element_located((By.NAME, "lower")))
        upper_input = wait.until(EC.presence_of_element_located((By.NAME, "upper")))
        
        lower_input.clear()
        lower_input.send_keys("50")
        upper_input.clear()
        upper_input.send_keys("50")
        
        submit_btn = driver.find_element(By.XPATH, "//button[@type='submit' and contains(text(), 'Submit')]")
        submit_btn.click()
        
        # 5. 최종 결과 페이지에서 p-value 추출
        info_span = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//span[contains(@class, 'info') and contains(text(), 'Logrank p-value')]")
        ))
        match = re.search(r"p-value=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", info_span.text)
        p_val = float(match.group(1)) if match else None
        
        # 6. CSV 데이터 다운로드 처리
        clear_temp_folder()
        
        # 알려주신 대로 <a> 태그가 아닌 <button> 태그로 XPath 수정
        download_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[@type='submit' and contains(text(), 'Click Here')]")
        ))
        download_btn.click()
        
        # 다운로드가 완료될 때까지 대기
        downloaded_file = wait_for_download()
        
        if downloaded_file:
            src_path = os.path.join(temp_download_dir, downloaded_file)
            new_filename = f"{cancer_type}_50_50_{gene}.csv"
            dst_path = os.path.join(target_dir, new_filename)
            
            shutil.move(src_path, dst_path)
            print(f"  └─ CSV 다운로드 완료 -> {dst_path}")
            return p_val
        else:
            print(f"  [오류] {gene}: CSV 다운로드 시간 초과")
            return p_val

    except Exception as e:
        print(f"  [오류] {gene} - {cancer_type}: 예기치 못한 에러 발생 -> {e}")
        return None

def main():
    gene_list = ["XRCC3"]
    significant_genes = []
    
    print("=== 웹 스크래핑 및 자동 다운로드를 시작합니다 ===\n")
    driver = setup_driver()
    
    try:
        for gene in gene_list:
            print(f"\n▶ 유전자 탐색 및 다운로드 중: {gene}")
            
            # LUAD 추출 및 파일 다운로드
            luad_p = download_gene_data(driver, gene, "LUAD")
            if luad_p is not None:
                print(f"  └─ p-value 추출 -> LUAD: {luad_p}")
            
            # LUSC 추출 및 파일 다운로드
            lusc_p = download_gene_data(driver, gene, "LUSC")
            if lusc_p is not None:
                print(f"  └─ p-value 추출 -> LUSC: {lusc_p}")
            
            # 둘 다 0.05 이하인 경우 기록
            if luad_p is not None and lusc_p is not None:
                if luad_p <= 0.05 and lusc_p <= 0.05:
                    significant_genes.append((gene, luad_p, lusc_p))
                    
    finally:
        driver.quit()
        if os.path.exists(temp_download_dir):
            shutil.rmtree(temp_download_dir)
        
    print("\n" + "="*50)
    print("  === 양쪽 모두 유의미한(p<=0.05) 유전자 목록 ===")
    print("="*50)
    for g in significant_genes:
        print(f"Gene: {g[0]} | LUAD: {g[1]} | LUSC: {g[2]}")
    print("="*50)
    print("스크래핑 완료! 이제 분석 파이프라인 코드를 실행하세요.")

if __name__ == "__main__":
    main()
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re

def setup_driver():
    """
    Selenium WebDriver를 설정합니다.
    디버깅을 위해 기본적으로 브라우저가 표시되며,
    운영 환경에서는 주석을 해제하여 Headless 모드로 사용할 수 있습니다.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') # Headless 모드 (최종 실행 시 주석 해제)
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(options=options)
    return driver

def get_p_value(driver, gene, cancer_type):
    """
    특정 유전자와 암종에 대한 Kaplan-Meier p-value를 추출합니다.
    '뒤로 가기'를 사용하지 않고 매번 메인 페이지에서 시작합니다.
    """
    wait = WebDriverWait(driver, 10)
    
    try:
        # 1. 메인 검색 페이지 접속
        driver.get("http://www.oncolnc.org/")
        
        # 2. 유전자 검색 칸 찾기 및 입력
        search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
        search_box.clear()
        search_box.send_keys(gene)
        search_box.send_keys(Keys.RETURN)
        
        # 3. 암종(cancer_type)에 해당하는 폼과 버튼 찾기
        # 입력한 암종(LUAD 또는 LUSC) 값을 가진 hidden input을 포함하는 폼의 submit 버튼
        xpath_btn = f"//form[input[@name='cancer' and @value='{cancer_type}']]//button[@type='submit']"
        
        # 유전자가 없거나 해당 암종의 데이터가 없을 경우를 대비한 대기
        btn = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_btn)))
        
        # 새 탭(target="_blank") 열림 방지를 위해 부모 form의 target 속성 제거
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
        
        # Submit 버튼 클릭
        submit_btn = driver.find_element(By.XPATH, "//button[@type='submit' and contains(text(), 'Submit')]")
        submit_btn.click()
        
        # 5. 최종 결과 페이지에서 p-value 추출
        info_span = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//span[contains(@class, 'info') and contains(text(), 'Logrank p-value')]")
        ))
        
        # 정규식을 통해 숫자값만 추출
        match = re.search(r"p-value=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", info_span.text)
        if match:
            return float(match.group(1))
        return None

    except TimeoutException:
        print(f"  [오류] {gene} - {cancer_type}: 페이지 로딩 지연 또는 요소를 찾을 수 없음 (Timeout)")
        return None
    except Exception as e:
        print(f"  [오류] {gene} - {cancer_type}: 예기치 못한 에러 발생 -> {e}")
        return None

def main():
    # 탐색할 유전자 리스트 (테스트용 예시)
    gene_list = ["tmem125"]
    significant_genes = []
    
    print("스크래핑을 시작합니다...\n")
    driver = setup_driver()
    
    try:
        for gene in gene_list:
            print(f"▶ 유전자 탐색 중: {gene}")
            
            # LUAD와 LUSC 각각 독립적인 세션 흐름으로 추출
            luad_p = get_p_value(driver, gene, "LUAD")
            lusc_p = get_p_value(driver, gene, "LUSC")
            
            print(f"  └─ 추출 완료 -> LUAD: {luad_p}, LUSC: {lusc_p}")
            
            # 두 값이 모두 정상적으로 추출되었고, 둘 다 0.05 이하인 경우
            if luad_p is not None and lusc_p is not None:
                if luad_p <= 0.05 and lusc_p <= 0.05:
                    significant_genes.append((gene, luad_p, lusc_p))
                    
    finally:
        driver.quit()
        
    # 결과 콘솔 출력
    print("\n" + "="*40)
    print("  === 유의미한 유전자 추출 결과 ===")
    print("="*40)
    print("significant_genes =", significant_genes)
    print("="*40)

if __name__ == "__main__":
    main()
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
target1 = "READ"
target2 = "COAD"

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
        # download_btn = wait.until(EC.element_to_be_clickable(
        #     (By.XPATH, "//button[@type='submit' and contains(text(), 'Click Here')]")
        # ))
        # download_btn.click()
        
        # 다운로드가 완료될 때까지 대기
        downloaded_file = None #wait_for_download()
        
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
    gene_list = [
    
    # 201-300: Extracellular Matrix, Adhesion, & EMT
    "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6", "COL5A1",
    "COL5A2", "COL6A1", "COL6A2", "COL6A3", "COL7A1", "COL8A1", "COL8A2", "COL9A1", "COL9A2", "COL9A3",
    "MMP1", "MMP3", "MMP7", "MMP10", "MMP11", "MMP12", "MMP13", "MMP14", "MMP15", "MMP16",
    "TIMP3", "TIMP4", "ADAM10", "ADAM17", "ADAMTS1", "ADAMTS4", "ADAMTS5", "ITGA1", "ITGA2", "ITGA3",
    "ITGA4", "ITGA5", "ITGA6", "ITGAV", "ITGB1", "ITGB3", "ITGB4", "ITGB5", "ITGB6", "ITGB8",
    "LAMB1", "LAMB3", "LAMC1", "LAMC2", "FN1", "VTN", "SPP1", "BGN", "DCN", "LUM",
    "VCAN", "ACAN", "NCAN", "TNC", "POSTN", "THBS1", "THBS2", "COMP", "SPARC", "CTGF",
    "CYR61", "NOV", "WISP1", "WISP2", "WISP3", "LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4",
    "CD44", "MCAM", "ALCAM", "NCAM1", "ICAM1", "VCAM1", "PECAM1", "SELE", "SELL", "SELP",
    "CDH3", "CDH5", "CDH11", "CDH13", "CDH17", "OCLN", "TJP1", "TJP2", "TJP3", "CLDN1",
    
    # 301-400: Immunology, Cytokines, Receptors, & Tumor Microenvironment
    "CLDN2", "CLDN3", "CLDN4", "CLDN7", "CLDN18", "CD274", "PDCD1", "CTLA4", "CD80", "CD86",
    "LAG3", "HAVCR2", "TIGIT", "IDO1", "CD276", "VTCN1", "BTLA", "CD40", "CD40LG", "CD28",
    "IL1A", "IL1B", "IL1RN", "IL2", "IL3", "IL4", "IL5", "IL7", "IL9", "IL10",
    "IL11", "IL12A", "IL12B", "IL13", "IL15", "IL17A", "IL17F", "IL18", "IL21", "IL22",
    "IL23A", "IL27", "IL33", "TNF", "TNFRSF1A", "TNFRSF1B", "FAS", "FASLG", "TRAIL", "TNFSF10",
    "IFNG", "IFNA1", "IFNB1", "CXCL1", "CXCL2", "CXCL3", "CXCL9", "CXCL10", "CXCL11", "CXCL13",
    "CCL2", "CCL3", "CCL4", "CCL5", "CCL7", "CCL19", "CCL20", "CCL21", "CCL22", "CCR2",
    "CCR4", "CCR5", "CCR6", "CCR7", "CXCR1", "CXCR2", "CXCR3", "CSF1", "CSF2", "CSF3",
    "CSF1R", "CD4", "CD8A", "CD8B", "CD3E", "CD3D", "CD3G", "CD14", "CD19", "CD20",
    "MS4A1", "CD33", "CD68", "CD163", "ITGAM", "ITGAX", "NCAM2", "FCGR3A", "NKG2D", "KLRC1",
    
    # 401-500: Metabolism, Solute Carriers, & ABC Transporters
    "SLC2A1", "SLC2A2", "SLC2A3", "SLC2A4", "SLC7A5", "SLC7A11", "SLC1A5", "SLC16A1", "SLC16A3", "SLC38A1",
    "SLC38A2", "SLC39A14", "SLC39A8", "SLC11A2", "SLC40A1", "SLC12A2", "SLC9A1", "ABCG2", "ABCB1", "ABCC1",
    "ABCC2", "ABCC3", "ABCA1", "ABCG1", "HK1", "HK2", "PFKP", "PFKM", "PFKL", "ALDOA",
    "ALDOB", "ALDOC", "TPI1", "GAPDH", "PGK1", "PGAM1", "ENO1", "ENO2", "PKM", "LDHA",
    "LDHB", "PDHA1", "PDHB", "PDHX", "PDK1", "PDK2", "PDK3", "PDK4", "CS", "ACO2",
    "IDH1", "IDH2", "IDH3A", "OGDH", "SUCLG1", "SUCLG2", "SDHA", "SDHB", "SDHC", "SDHD",
    "FH", "MDH1", "MDH2", "FASN", "ACACA", "ACACB", "SCD", "SREBF1", "SREBF2", "HMGCR",
    "HMGCS1", "FDPS", "SQLE", "LSS", "ACAT1", "ACAT2", "CPT1A", "CPT1B", "CPT1C", "CPT2",
    "ACADVL", "ACADM", "ACADS", "HADHA", "HADHB", "GLS", "GLS2", "GLUD1", "GOT1", "GOT2",
    "GPT", "GPT2", "ASS1", "ASL", "ARG1", "ARG2", "OTC", "CPS1", "CAD", "UMPS",
    
    # 501-600: Epigenetics, Chromatin Modifiers, & Transcription Factors
    "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2", "TET3", "EZH2", "SUZ12", "EED", "BMI1",
    "CBX2", "CBX4", "CBX7", "CBX8", "RING1", "RNF2", "KMT2A", "KMT2B", "KMT2C", "KMT2D",
    "SETD2", "NSD1", "NSD2", "NSD3", "DOT1L", "KDM1A", "KDM2A", "KDM2B", "KDM3A", "KDM4A",
    "KDM4B", "KDM4C", "KDM5A", "KDM5B", "KDM5C", "KDM6A", "KDM6B", "HDAC1", "HDAC2", "HDAC3",
    "HDAC4", "HDAC5", "HDAC6", "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11", "SIRT1", "SIRT2",
    "SIRT3", "SIRT4", "SIRT5", "SIRT6", "SIRT7", "HAT1", "KAT2A", "KAT2B", "KAT5", "EP300",
    "CREBBP", "BRD2", "BRD3", "BRD4", "BRDT", "SMARCA2", "SMARCA4", "SMARCB1", "SMARCC1", "SMARCC2",
    "ARID1A", "ARID1B", "ARID2", "PBRM1", "CHD1", "CHD2", "CHD3", "CHD4", "CHD5", "CHD6",
    "FOXA1", "FOXA2", "FOXA3", "FOXC1", "FOXC2", "FOXF1", "FOXF2", "FOXJ1", "FOXM1", "FOXO1",
    "FOXO3", "FOXO4", "FOXP1", "FOXP2", "FOXP3", "FOXP4", "HOXA1", "HOXA2", "HOXA3", "HOXA4",
    
    # 601-700: More Transcription Factors, Zinc Fingers, & RNA processing
    "HOXA5", "HOXA6", "HOXA7", "HOXA9", "HOXA10", "HOXA11", "HOXA13", "HOXB1", "HOXB2", "HOXB3",
    "HOXB4", "HOXB5", "HOXB6", "HOXB7", "HOXB8", "HOXB9", "HOXB13", "HOXC4", "HOXC5", "HOXC6",
    "HOXC8", "HOXC9", "HOXC10", "HOXC11", "HOXC12", "HOXC13", "HOXD1", "HOXD3", "HOXD4", "HOXD8",
    "HOXD9", "HOXD10", "HOXD11", "HOXD12", "HOXD13", "GATA1", "GATA2", "GATA3", "GATA4", "GATA5",
    "GATA6", "STAT1", "STAT2", "STAT4", "STAT5A", "STAT5B", "STAT6", "RELA", "RELB", "NFKB1",
    "NFKB2", "JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2", "ATF1", "ATF2",
    "ATF3", "ATF4", "ATF6", "CREB1", "CREM", "SP1", "SP2", "SP3", "SP4", "ZNF10",
    "ZNF14", "ZNF16", "ZNF18", "ZNF22", "ZNF24", "ZNF28", "ZNF30", "ZNF34", "ZNF38", "ZNF41",
    "ZNF43", "ZNF45", "ZNF48", "ZNF58", "ZNF71", "ZNF74", "ZNF75", "ZNF85", "ZNF90", "ZNF91",
    "SRSF1", "SRSF2", "SRSF3", "SRSF4", "SRSF5", "SRSF6", "SRSF7", "SRSF8", "SRSF9", "SRSF10",
    
    # 701-800: Ribosomal Proteins, Translation, & Proteasome
    "HNRNPA1", "HNRNPA2B1", "HNRNPK", "HNRNPL", "HNRNPU", "RPL3", "RPL4", "RPL5", "RPL6", "RPL7",
    "RPL8", "RPL9", "RPL10", "RPL11", "RPL12", "RPL13", "RPL14", "RPL15", "RPL18", "RPL19",
    "RPL21", "RPL22", "RPL23", "RPL24", "RPL26", "RPL27", "RPL28", "RPL29", "RPL30", "RPL31",
    "RPL32", "RPL34", "RPL35", "RPL36", "RPL37", "RPL38", "RPL39", "RPL40", "RPS2", "RPS3",
    "RPS4X", "RPS4Y1", "RPS5", "RPS6", "RPS7", "RPS8", "RPS9", "RPS10", "RPS11", "RPS12",
    "RPS13", "RPS14", "RPS15", "RPS16", "RPS17", "RPS18", "RPS19", "RPS20", "RPS21", "RPS23",
    "RPS24", "RPS25", "RPS26", "RPS27", "RPS28", "RPS29", "EIF2S1", "EIF2S2", "EIF2S3", "EIF3A",
    "EIF3B", "EIF3C", "EIF3D", "EIF3E", "EIF4A1", "EIF4E", "EIF4G1", "EEF1A1", "EEF1A2", "EEF2",
    "PSMA1", "PSMA2", "PSMA3", "PSMA4", "PSMA5", "PSMA6", "PSMA7", "PSMB1", "PSMB2", "PSMB3",
    "PSMB4", "PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "PSMC1", "PSMC2", "PSMC3",
    
    # 801-900: Cytoskeleton, Motor Proteins, & Intracellular Trafficking
    "ACTB", "ACTG1", "ACTA1", "ACTA2", "ACTC1", "TUBA1A", "TUBA1B", "TUBA1C", "TUBA4A", "TUBB",
    "TUBB2A", "TUBB2B", "TUBB3", "TUBB4A", "TUBB4B", "TUBG1", "VIM", "DES", "GFAP", "KRT1",
    "KRT5", "KRT7", "KRT8", "KRT14", "KRT18", "KRT19", "KRT20", "MAP2", "MAP4", "MAPT",
    "MYH9", "MYH10", "MYL9", "MYL12A", "MYL12B", "MYO5A", "MYO5B", "MYO5C", "MYO6", "MYO7A",
    "KIF1A", "KIF1B", "KIF2A", "KIF2C", "KIF3A", "KIF3B", "KIF4A", "KIF5B", "KIF5C", "KIF12",
    "KIF14", "KIF15", "KIF18A", "KIF20B", "KIF22", "KIF24", "DYNC1H1", "DYNC1I1", "DYNC1I2", "DYNC1LI1",
    "RAB1A", "RAB1B", "RAB2A", "RAB3A", "RAB4A", "RAB5A", "RAB5B", "RAB5C", "RAB7A", "RAB8A",
    "RAB9A", "RAB10", "RAB11A", "RAB11B", "RAB14", "RAB25", "RAB27A", "RAB27B", "ARF1", "ARF3",
    "ARF4", "ARF5", "ARF6", "RAC1", "RAC2", "RAC3", "RHOA", "RHOB", "RHOC", "CDC42",
    "SEC23A", "SEC24A", "SEC24B", "SEC24C", "SEC24D", "SEC31A", "COPA", "COPB1", "COPG1", "ARPC1A",
    
    # 901-1000: Apoptosis, Heat Shock, & Assorted Signaling
    "ARPC1B", "ARPC2", "ARPC3", "ARPC4", "ARPC5", "WAS", "WASL", "WAVE1", "WAVE2", "WAVE3",
    "HSPA1A", "HSPA1B", "HSPA2", "HSPA4", "HSPA5", "HSPA8", "HSPA9", "HSPB1", "HSPD1", "HSPE1",
    "HSP90AA1", "HSP90AB1", "HSP90B1", "DNAJA1", "DNAJA2", "DNAJB1", "DNAJC1", "DNAJC3", "BAG1", "BAG3",
    "BCL2L1", "BCL2L2", "BCL2L11", "MCL1", "PMAIP1", "BBC3", "BID", "BAD", "BIK", "HRK",
    "CASP1", "CASP2", "CASP4", "CASP5", "CASP6", "CASP7", "CASP10", "CASP14", "APAF1", "DIABLO",
    "HTRA2", "XIAP", "BIRC2", "BIRC3", "BIRC5", "BIRC6", "FLT3", "KIT", "PDGFRA", "PDGFRB",
    "RET", "ROS1", "ALK", "NTRK1", "NTRK2", "NTRK3", "EPHB2", "EPHB4", "EPHA2", "EPHA3",
    "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1", "JAG2", "DLL1", "DLL3", "DLL4", "HES1",
    "HEY1", "HEY2", "NUMB", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2", "GLI3", "SUFU",
    "YAP1", "TAZ", "LATS1", "LATS2", "MST1", "MST2", "SAV1", "MOB1A", "MOB1B", "TEAD1"
]
    significant_genes = []
    
    print("=== 웹 스크래핑 및 자동 다운로드를 시작합니다 ===\n")
    driver = setup_driver()
    
    try:
        for gene in gene_list:
            print(f"\n▶ 유전자 탐색 및 다운로드 중: {gene}")
            
            
            # target1 추출 및 파일 다운로드
            target1_p = download_gene_data(driver, gene, f"{target1}")
            if target1_p is not None:
                print(f"  └─ p-value 추출 -> {target1}: {target1_p}")
            
            # target2 추출 및 파일 다운로드
            target2_p = download_gene_data(driver, gene, f"{target2}")
            if target2_p is not None:
                print(f"  └─ p-value 추출 -> {target2}: {target2_p}")
            
            # 둘 다 0.05 이하인 경우 기록
            if target1_p is not None and target2_p is not None:
                if target1_p <= 0.05 and target2_p <= 0.05:
                    significant_genes.append((gene, target1_p, target2_p))
                    
    finally:
        driver.quit()
        if os.path.exists(temp_download_dir):
            shutil.rmtree(temp_download_dir)
        
    print("\n" + "="*50)
    print("  === 양쪽 모두 유의미한(p<=0.05) 유전자 목록 ===")
    print("="*50)
    for g in significant_genes:
        print(f"Gene: {g[0]} | {target1}: {g[1]} | {target2}: {g[2]}")
    print("="*50)
    print("스크래핑 완료! 이제 분석 파이프라인 코드를 실행하세요.")

if __name__ == "__main__":
    main()
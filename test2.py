import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# =====================================================================
# 디렉토리 설정
# =====================================================================
try:
    super_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    super_dir = os.path.abspath(".")

temp_download_dir = os.path.join(super_dir, "temp_download")
os.makedirs(temp_download_dir, exist_ok=True)

def debug_setup_driver():
    print("\n--- [디버그 1] ChromeOptions 설정 시작 ---")
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    prefs = {
        "download.default_directory": temp_download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)
    print("--- [디버그 1 완료] ChromeOptions 설정 완료 ---")

    # 1. webdriver-manager 사용 여부 확인
    print("\n--- [디버그 2] webdriver-manager 라이브러리 체크 ---")
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print(">> webdriver-manager 가 설치되어 있습니다. 자동 다운로더를 사용합니다.")
        
        print(">> ChromeDriver 다운로드/경로 확인 중... (여기서 멈추면 네트워크/방화벽 문제)")
        driver_path = ChromeDriverManager().install()
        print(f">> ChromeDriver 위치: {driver_path}")
        
        service = Service(driver_path)
        
        print("\n--- [디버그 3] Chrome 브라우저 실행 시도 중... ---")
        driver = webdriver.Chrome(service=service, options=options)
        print(">> [성공] Chrome 브라우저가 정상적으로 열렸습니다!")
        return driver

    except ImportError:
        print(">> [안내] webdriver-manager가 설치되어 있지 않습니다.")
        print(">> Selenium 기본 매니저로 실행을 시도합니다.")
        
        print("\n--- [디버그 3] Chrome 브라우저 실행 시도 중... (여기서 멈추면 프로세스 충돌 또는 Selenium 매니저 먹통) ---")
        try:
            driver = webdriver.Chrome(options=options)
            print(">> [성공] Chrome 브라우저가 정상적으로 열렸습니다!")
            return driver
        except Exception as e:
            print(f"\n[에러 발생] Chrome 실행 실패: {e}")
            return None

    except Exception as e:
        print(f"\n[에러 발생] 예기치 못한 에러: {e}")
        return None

if __name__ == "__main__":
    print("=== Selenium 디버깅 시작 ===")
    
    start_time = time.time()
    driver = debug_setup_driver()
    
    if driver:
        print(f"\n소요 시간: {round(time.time() - start_time, 2)}초")
        print("5초 후 브라우저를 종료합니다...")
        time.sleep(5)
        driver.quit()
        print("=== 디버깅 종료 ===")
    else:
        print("\n=== 드라이버 로드 실패 ===")
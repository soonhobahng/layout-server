#!/usr/bin/env python3
"""
SSL 인증서 문제 해결 스크립트
"""

import subprocess
import sys
import ssl
import urllib.request
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_ssl_certificates():
    """SSL 인증서 문제 해결"""
    
    print("=== SSL 인증서 문제 해결 중 ===")
    
    # 1. certifi 업데이트
    try:
        print("1. certifi 패키지 업데이트...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "certifi"], check=True)
        print("✓ certifi 업데이트 완료")
    except subprocess.CalledProcessError as e:
        print(f"✗ certifi 업데이트 실패: {e}")
    
    # 2. macOS 인증서 설치 (macOS인 경우)
    if sys.platform == "darwin":
        try:
            print("2. macOS 인증서 설치...")
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            cert_install_path = f"/Applications/Python {python_version}/Install Certificates.command"
            
            if Path(cert_install_path).exists():
                subprocess.run([cert_install_path], check=True)
                print("✓ macOS 인증서 설치 완료")
            else:
                print("⚠ macOS 인증서 설치 스크립트를 찾을 수 없음")
        except Exception as e:
            print(f"✗ macOS 인증서 설치 실패: {e}")
    
    # 3. 환경 변수 설정
    print("3. 환경 변수 설정 방법:")
    print("   export PYTHONHTTPSVERIFY=0")
    print("   export SSL_CERT_FILE=$(python -m certifi)")
    print("   export REQUESTS_CA_BUNDLE=$(python -m certifi)")
    
    # 4. 수동 다운로드 방법
    print("\n4. 수동 모델 다운로드 방법:")
    print("   모델이 자동 다운로드되지 않는 경우, 다음 링크에서 수동 다운로드:")
    print("   - Config: https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1")
    print("   - Model: https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1")
    
    model_dir = Path.home() / '.layoutparser' / 'models' / 'faster_rcnn_R_50_FPN_3x'
    print(f"   다운로드한 파일을 다음 경로에 저장: {model_dir}")
    
    # 5. 연결 테스트
    print("\n5. 연결 테스트...")
    test_url = "https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1"
    
    try:
        # SSL 검증 비활성화로 테스트
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(test_url, context=ssl_context, timeout=10) as response:
            if response.status == 200:
                print("✓ SSL 검증 비활성화로 연결 성공")
                return True
            else:
                print(f"✗ 연결 실패: HTTP {response.status}")
    except Exception as e:
        print(f"✗ 연결 테스트 실패: {e}")
    
    return False

def download_models_manually():
    """모델 파일 수동 다운로드"""
    
    print("\n=== 모델 수동 다운로드 ===")
    
    model_urls = {
        'config.yml': 'https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1',
        'model_final.pth': 'https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1'
    }
    
    model_dir = Path.home() / '.layoutparser' / 'models' / 'faster_rcnn_R_50_FPN_3x'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"모델 저장 경로: {model_dir}")
    
    # SSL 컨텍스트 설정
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for filename, url in model_urls.items():
        file_path = model_dir / filename
        
        if file_path.exists():
            print(f"✓ {filename} 이미 존재함")
            continue
            
        try:
            print(f"다운로드 중: {filename}...")
            
            with urllib.request.urlopen(url, context=ssl_context, timeout=30) as response:
                with open(file_path, 'wb') as f:
                    f.write(response.read())
            
            print(f"✓ {filename} 다운로드 완료")
            
        except Exception as e:
            print(f"✗ {filename} 다운로드 실패: {e}")
            print(f"   수동으로 다음 URL에서 다운로드하여 {file_path}에 저장하세요:")
            print(f"   {url}")

def main():
    print("Layout Parser SSL 문제 해결 도구")
    print("=" * 50)
    
    # 1. SSL 인증서 문제 해결
    fix_ssl_certificates()
    
    print("\n" + "=" * 50)
    
    # 2. 모델 수동 다운로드 시도
    try:
        download_models_manually()
    except Exception as e:
        print(f"모델 다운로드 실패: {e}")
        print("\n대안 해결책:")
        print("1. VPN 연결 해제")
        print("2. 회사 네트워크가 아닌 다른 네트워크 사용")
        print("3. 프록시 설정 확인")
        print("4. 방화벽 설정 확인")
    
    print("\n" + "=" * 50)
    print("해결책 요약:")
    print("1. pip install --upgrade certifi")
    print("2. export PYTHONHTTPSVERIFY=0")
    print("3. 모델 파일 수동 다운로드")
    print("4. 네트워크 환경 확인")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
빠른 모델 다운로드 및 설정 스크립트
"""

import os
import ssl
import urllib.request
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_files():
    """모델 파일을 수동으로 다운로드"""
    
    # 모델 저장 경로
    model_dir = Path.home() / '.layoutparser' / 'models' / 'faster_rcnn_R_50_FPN_3x'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"모델 저장 경로: {model_dir}")
    
    # 파일 URL과 경로
    files_to_download = {
        'config.yml': 'https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1',
        'model_final.pth': 'https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1'
    }
    
    # SSL 설정
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for filename, url in files_to_download.items():
        file_path = model_dir / filename
        
        if file_path.exists():
            print(f"✓ {filename} 이미 존재")
            continue
        
        print(f"📥 {filename} 다운로드 중...")
        
        try:
            # 요청 헤더 설정
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(request, context=ssl_context, timeout=60) as response:
                with open(file_path, 'wb') as f:
                    f.write(response.read())
            
            print(f"✅ {filename} 다운로드 완료")
            
        except Exception as e:
            print(f"❌ {filename} 다운로드 실패: {e}")
            print(f"수동 다운로드: {url}")
    
    return model_dir

def setup_environment():
    """환경 설정"""
    
    print("🔧 환경 설정 중...")
    
    # SSL 환경 변수 설정
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    # certifi 경로 설정
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        print(f"✓ SSL 인증서 경로 설정: {certifi.where()}")
    except ImportError:
        print("⚠ certifi 패키지가 설치되지 않음")
    
    print("✓ 환경 설정 완료")

def test_model_loading():
    """모델 로딩 테스트"""
    
    print("🧪 모델 로딩 테스트...")
    
    try:
        import layoutparser as lp
        
        # 모델 로딩 시도
        model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            model_path='/Users/soonhobahng/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth?dl=1',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7]
        )
        
        print("✅ 실제 모델 로딩 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def main():
    print("=" * 60)
    print("Layout Parser 모델 설정 도구")
    print("=" * 60)
    
    # 1. 환경 설정
    setup_environment()
    
    # 2. 모델 파일 다운로드
    model_dir = download_model_files()
    
    # 3. 모델 로딩 테스트
    success = test_model_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 설정 완료! 실제 레이아웃 모델을 사용할 수 있습니다.")
    else:
        print("⚠ 모델 로딩 실패. Mock 모델이 사용됩니다.")
        print("\n대안:")
        print("1. VPN 연결 해제")
        print("2. 방화벽 설정 확인")
        print("3. pip install --upgrade certifi")
        print("4. 서버 재시작")
    
    print(f"\n모델 파일 위치: {model_dir}")
    print("서버 재시작 후 테스트해주세요.")

if __name__ == "__main__":
    main()
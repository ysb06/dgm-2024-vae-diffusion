# Hybrid DiffuseVAE 기말 프로젝트

## 개발 환경 설정

**아래 개발 환경 설정과 관련된 부분은 참고만 하시고 편하신대로 세팅하셔도 무방합니다.**

대부분 가상환경 구성과 관련되어 있으므로 가상환경 사용하지 않으면 맨 아래 `pip install -e .`실행과 CUDA 사용 시, CUDA 버전 PyTorch만 설치하셔도 무방합니다.

### PDM 사용 시 (PDM이 관리하는 venv 사용 시)

현재 Baseline 프로젝트 구성 시, PDM을 사용했습니다. 다만, 이것은 패키징 보조 도구라서 꼭 사용하지 않아도 무방합니다.

PDM 설치 관련해서는 아래 사이트들 참조 바랍니다.
- https://pdm-project.org/en/latest/
- https://otzslayer.github.io/python/2023/06/28/pdm-python-dependency-manager.html

CUDA 버전 PyTorch 사용을 위해서는 다음을 필수로 실행합니다.

```
pdm config venv.with_pip True
```

아래 실행 후 pdm이 관리하는 venv 생성 및 주요 패키지들을 설치합니다.
```
pdm update
pdm venv activate
```

(윈도우 한정) 이후, CUDA 버전 PyTorch 사용을 위해서는 다음과 같이 pip를 사용해 패키지 업데이트 진행합니다. pdm add, pdm update 등 실행 후 초기화 되므로 이 경우에 학습이나 추론 전에 아래를 계속 실행 후 코드 실행해야 합니다.

아래는 CUDA 12.1 버전의 PyTorch 최신 버전 설치 예시입니다.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### PDM 사용하지 않고 venv 가상 환경 사용 시

아래를 진행하기 전에 현재 폴더가 프로젝트 최상단 폴더에 있는지 확인합니다.

1. 가상환경 생성
    ```
    python -m venv .venv
    ```

2. 가상환경 활성화

    리눅스 / Mac 기준,
    ```
    source .venv/bin/activate
    ```

3. 필요 패키지 설치

    개발 중이므로 다음과 같이 Editable하게 설치합니다.
    ```
    pip install -e .
    ```

## Baseline 코드 실행

원래 코드를 왠만하면 유지하려고 했지만 여러가지 이유로 코드가 수정되고 실행방법이 변경되었습니다. 기존 원 소스에서 지원한 스크립트 실행은 동작하지 않습니다. 현재는 실행을 위해서는 파이썬 코드(`./src/baseline/__main__.py`)를 수정해야 합니다. 다만, 스크립트 자체는 `./src/baseline/scripts`에 참고용으로 남겨 두었습니다.

데이터셋 다운로드 스크립트(`./download.sh`)는 아마 동작할 것으로 예상됩니다.

1. 필요한 경우 config 수정

    데이터셋을 변경하거나 테스트로 진행할 경우 `./src/baseline/configs/config.yaml`의 dataset 속성의 경로를 수정합니다.

    예시) celeba 테스트를 수행하는 것으로 변경할 경우,
    ```yaml
    defaults:
        - dataset: celeba64/test
    ```

2. 실행 코드 수정
    `./src/baseline/__main__.py`의 내용을 수정하여 학습할 모델을 변경하거나 추론으로 변경합니다.

3. 코드 실행
    - 터미널에서 실행 시, (프로젝트 최상단 폴더에서 실행)
        ```
        python -m baseline
        ```
        가상환경이 활성화 되어 있어야 하며, 가상환경에서 `pip install -e .`를 통해 현재 프로젝트가 설치되어 있어야 합니다.

    - VSCode에서 실행 시,
        
        디버그 탭에서 Run Baseline 선택 후, 디버깅을 시작합니다.
    
    현재는 위와 같이 실행할 시, VAE 모델이 CIFAR10 데이터셋으로 학습되는 코드로 작성되어 있습니다.

## Notice

세세한 것들은 추후 같이 논의해 보아요. 이슈 있으면 언제든 카톡에서 이야기 해 봐요.
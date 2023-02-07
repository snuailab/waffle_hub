![header](https://capsule-render.vercel.app/api?type=soft&color=00a2bf&fontColor=3A3A3C&height=120&section=header&text=AutoCare%20CI&fontAlign=30&desc=snuailab&descAlign=90&descAlignY=80&descSize=40&fontSize=80)

## Autocare CI Template

![python](https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white)

# Overview
autocare CI 레포는 다음 작업들에 대한 템플릿을 제공합니다.
- [github issue 템플릿을 지정합니다.](#issue-template)
- [commit 시에 지정된 동작이 실행됩니다.](#on-commit)
- [pull request 시에 지정된 동작이 실행됩니다.](#on-pull-request)
- [주기적인 dependency 관리를 수행합니다.](#on-schedule)

# How to use
1. 새로운 프로젝트 생성 시 해당 레포를 Template 으로 지정하세요
    <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/067a1c71-9307-43d4-ac6e-d76927f62650/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230201T071121Z&X-Amz-Expires=86400&X-Amz-Signature=f50a2a57e57257652462d96d432ffc84883c4f3170dc5ce3e9d5a84ac21c6eab&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject" width="70%"/>
    <img src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5017e0ef-7fe9-4776-a08c-6fd0bb698625/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230201T071620Z&X-Amz-Expires=86400&X-Amz-Signature=8aeb1dc3dd6efa9a918c4a451f19eac9ebbb8c38098791d0543b204a9886c826&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject" width="70%"/>
2. .pre-commit-config.yaml 파일의 python 버전을 수정합니다.
    ```python
    default_language_version:
    python: python{version}
    ```
3. **레포 클론시에 다음 작업을 꼭! 수행합니다.**
    ```python
    pip install pre-commit
    pre-commit install
    ```
4. 이외 개발 workflow 는 [SNUAILAB Workflow](https://www.notion.so/snuailab/Workflow-fe17516d921c4d4b92d2eefca219d140)를 따릅니다.

# Specifications
## Issue template
이슈 템플릿을 지정합니다.
- [bug-report.yaml](.github/ISSUE_TEMPLATE/bug-report.yaml)
- [feature-request.yaml](.github/ISSUE_TEMPLATE/feature-request.yaml)
- [question.yaml](.github/ISSUE_TEMPLATE/question.yaml)
- [config.yaml](.github/ISSUE_TEMPLATE/config.yaml)

## on Commit
commit 시에 실행되는 동작으로 다음 작업이 수행됩니다. 자세한 내용은 [.pre-commit-config.yaml](.pre-commit-config.yaml) 파일 참고
- pep8
- codespell
- black
- isort
- pyupgrade
- etc.

## on Pull Request
pr 시에 실행되는 동작으로 다음 작업들이 수행됩니다.
- [ci.yaml](.github/workflows/ci.yaml)

## on Schedule
schedule 되어 실행되는 동작으로 다음 작업들이 수행됩니다.
- [dependabot.yaml](.github/dependabot.yaml)

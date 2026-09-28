# Git 업로드용 백업

실행 중인 discos_minji와 분리한 시점 스냅샷이다. 원본 파일은 수정/삭제하지 않았다.
코드·설정·문서·작은 JSON/CSV·그래프를 포함한다. 제외 항목과 SHA-256은
`backup_meta/manifest.json`에서 확인한다. 파일당 기본 10MiB 제한이 있다.

가상환경, 컴파일 바이너리, 모델, 원시 .log/.jsonl, 실행 중 실험 디렉터리는 제외했다.
DAOS 서버의 실제 KV 데이터 및 /opt 네이티브 라이브러리도 포함하지 않는다.
그러므로 전체 서버/원시 실험 데이터 백업이 아니며, 복구 후 바로 실행 가능하다는 뜻도 아니다.
스크립트의 /root/discos_minji, /opt 및 HF 캐시 경로는 새 환경에 맞춰 확인해야 한다.

`backup_meta/python_packages.json`은 설치 버전 목록이며 재현이 검증된 lock 파일은 아니다.
벤치마크 원본은 `backup_meta/upstreams.json`의 URL/commit으로 복원한다.
같은 commit에 해당 .patch를 적용한다. 비밀정보가 들어갈 수 있는 api_config.json과
원본 저장소의 untracked .bak는 제외했다. API 설정은 별도로 복원한다.
libdaosgdr.so는 설치된 GPU 지원 DAOS 스택을 준비한 뒤 Makefile로 다시 빌드한다.

## 업로드

초기 커밋은 자동 스냅샷임을 나타내는 Local Backup <backup@localhost> 명의다.
서버 경로·실험 내용·생성 답변이 들어 있으므로 비공개 저장소를 권장한다.
토큰 패턴 검사는 완전한 비밀정보/개인정보 검사를 대체하지 않으므로 공개 전 검토한다.
비밀정보를 URL에 넣지 말고 SSH 또는 Git 자격증명 관리자를 사용한다.

빈 원격 저장소를 준비한 뒤 이 디렉터리에서:

```bash
git remote add origin <저장소_URL>
git push -u origin main
```

원격에 기존 이력이 있으면 강제 push하지 말고 먼저 통합 방법을 확인한다.
현재 작업의 자동 동기화나 자동 업로드는 하지 않는다. 실행 중이던 실험이 끝난 뒤
원본의 prepare_git_backup.py --output <새_백업_절대경로>로 새 스냅샷을 만들 수 있다.
다음 스냅샷은 별도 Git 이력이므로 기존 원격을 덮어쓰지 말고 검토 후 기존 백업에 반영한다.

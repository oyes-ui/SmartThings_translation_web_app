"""Render existing review decisions without inventing candidates or approving edits."""
from pathlib import Path
from workbook_contract import atomic_json, atomic_text, digest

BEGIN = "<!-- st-workflow:begin -->"
END = "<!-- st-workflow:end -->"


def managed_write(path: Path, generated: str):
    if BEGIN in generated or END in generated:
        raise ValueError("Reserved marker in generated report")
    block = BEGIN + "\n" + generated.rstrip() + "\n" + END
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old.count(BEGIN) != 1 or old.count(END) != 1 or old.index(BEGIN) > old.index(END):
            raise ValueError(f"Existing note has no unique managed block; preserved: {path}")
        content = old[:old.index(BEGIN)] + block + old[old.index(END) + len(END):]
    else:
        content = block + "\n\n## 사용자 메모\n\n"
    atomic_text(path, content)


def escape(value):
    return str(value if value is not None else "").replace("|", "&#124;").replace("\n", "<br>")


def publish(workflow: dict, root: Path):
    from workbook_batch import read, settings_for
    # Stable workflow-specific name prevents unrelated runs overwriting each other.
    name = "approval-" + digest(str(root.resolve()))[:12]
    rows, changes, details = [], [], []
    for job in workflow["jobs"]:
        try:
            settings = settings_for(job)
        except Exception:
            rows.append(f"| {job['job_id']} | — | settings_error | 0 | 입력 변경 | — |")
            continue
        status, count, queue_count = job["state"], 0, 0
        link = "—"
        report_error = None
        start_changes, start_details = len(changes), len(details)
        try:
            if job.get("review_manifest"):
                batch = read(job["review_manifest"])
                for language in batch["jobs"]:
                    summary = Path(language["work_dir"]) / "final_summary.json"
                    if not summary.exists():
                        continue
                    result = read(summary)
                    manifest = read(result["manifest"])
                    detail_name = f"review-{name.removeprefix('approval-')}-{job['job_id']}.md"
                    source_md = Path(result["report"]).read_text(encoding="utf-8")
                    source_md += "\n\n## 승인 후보 식별자\n\n"
                    for candidate in manifest["changes"]:
                        source_md += (f"- `{candidate.get('finding_id', '')}` · {escape(candidate.get('cell'))}: "
                                      f"{escape(candidate.get('before'))} → {escape(candidate.get('after'))}\n")
                    queue_count = len(manifest.get("review_context", {}).get("human_review_queue", []))
                    if queue_count:
                        source_md += f"\n사용자 판단 필요: {queue_count}건\n"
                    details.append((detail_name, source_md))
                    link = f"[[{Path(detail_name).stem}]]"
                    if status != "completed":
                        continue
                    for change in manifest["changes"]:
                        changes.append({"job_id": job["job_id"], "story": settings["story"],
                                        "language": settings["sheet"], "settings": job["settings"],
                                        "source_manifest": result["manifest"], "change": change})
                        count += 1
        except (ValueError, OSError, KeyError, TypeError) as error:
            del changes[start_changes:]
            del details[start_details:]
            status, count, queue_count, link = "report_error", 0, 0, "—"
            report_error = str(error)
        note = "수정 후보 없음" if status == "completed" and count == 0 else report_error or job.get("error") or ""
        if queue_count:
            note = f"{note}; 사용자 판단 필요 {queue_count}건"
        rows.append("| " + " | ".join(map(escape, [settings["story"], settings["sheet"], status, count, note])) + " | " + link + " |")
    lines = ["# 통합 승인검토표", "", "기존 검수 결과와 원본 manifest의 승인 상태를 모은 문서입니다. 새 후보는 승인 대기이며 추가 검수 단계는 없습니다.", "",
             "| Story | 언어 | 상태 | 후보 수 | 참고 | 상세 검수 |", "| --- | --- | --- | --- | --- | --- |", *rows]
    last_group = None
    for entry in sorted(changes, key=lambda x: (str(x["story"]), x["language"], x["job_id"])):
        group = str(entry["story"]), entry["language"]
        if group != last_group:
            lines.extend(["", f"## {escape(group[0])} · {escape(group[1])}", "",
                          "| 작업 · 후보 ID | 셀 | 기존 | 제안 | 근거 | 판단 |",
                          "| --- | --- | --- | --- | --- | --- |"])
            last_group = group
        change = entry["change"]
        fields = [f"{entry['job_id']} · {change.get('finding_id', '')}", change.get("cell"),
                  change.get("before"), change.get("after"), change.get("reason", ""),
                  {"approved": "승인", "rejected": "거절", "pending_approval": "승인 대기"}.get(change.get("approval_status"), "승인 대기")]
        lines.append("| " + " | ".join(map(escape, fields)) + " |")
    markdown = "\n".join(lines) + "\n"
    report_dir = root / "reports"
    for filename, content in details:
        managed_write(report_dir / filename, content)
    managed_write(report_dir / f"{name}.md", markdown)
    mapping = {"schema_version": 1, "kind": "approval_index", "changes": changes,
               "jobs": [{"job_id": j["job_id"], "state": j["state"], "settings": j["settings"]} for j in workflow["jobs"]]}
    atomic_json(report_dir / f"{name}.json", mapping)
    published = {"report": str(report_dir / f"{name}.md"), "index": str(report_dir / f"{name}.json"),
                 "obsidian_status": "location_required"}
    if workflow.get("obsidian_dir"):
        vault = Path(workflow["obsidian_dir"])
        try:
            for filename, content in details:
                managed_write(vault / filename, content)
            managed_write(vault / f"{name}.md", markdown)
            published.update(obsidian_status="published", obsidian_report=str(vault / f"{name}.md"))
        except (ValueError, OSError) as error:
            published.update(obsidian_status="blocked_existing_note" if isinstance(error, ValueError) else "write_failed", error=str(error))
    return published

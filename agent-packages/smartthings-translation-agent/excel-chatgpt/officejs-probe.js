/*
 * SmartThings Excel Live — read-only Office.js capability probe
 *
 * 실행 전제: Microsoft Excel의 ChatGPT add-in이 연결된 live session에서만 실행한다.
 * 이 파일은 Excel.run/run_officejs가 허용한 코드 입력에 붙여 넣는 probe다.
 * 어떤 셀, workbook, table도 만들거나 수정하지 않는다.
 */

async function smartThingsOfficeJsProbe() {
  return Excel.run(async (context) => {
    const workbook = context.workbook;
    const worksheet = workbook.worksheets.getActiveWorksheet();
    const selection = workbook.getSelectedRange();
    const protection = worksheet.protection;

    worksheet.load(["name", "visibility"]);
    selection.load(["address", "values", "formulas", "text"]);
    protection.load(["protected"]);
    await context.sync();

    const firstValue = selection.values?.[0]?.[0] ?? null;
    const firstFormula = selection.formulas?.[0]?.[0] ?? null;
    const firstText = selection.text?.[0]?.[0] ?? null;
    const mergedState = firstValue === null && firstText !== null
      ? "unknown: inspect manually before writing"
      : "not assessed by read-only probe";

    return {
      probe: "smartthings-officejs-live-v1",
      writePerformed: false,
      target: {
        worksheet: worksheet.name,
        visibility: worksheet.visibility,
        selection: selection.address,
        protected: protection.protected,
      },
      cell: { firstValue, firstFormula, firstText, mergedState },
      capabilities: {
        readSelection: true,
        writeRange: "requires separately advertised live command and approval",
        partialRichText: "not asserted: Excel Office.js range APIs do not provide a verified SmartThings run-level contract",
        glossaryTable: "requires explicit CSV import plus a separately approved write step",
      },
      decision: "preview_only",
      fallback: "Use Delivery Python for glossary substring rich-text highlighting.",
    };
  });
}

smartThingsOfficeJsProbe();

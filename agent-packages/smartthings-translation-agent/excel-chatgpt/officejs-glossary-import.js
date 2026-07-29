/* User-pasted payload only. Never receives a local CSV path or secret. */
async function importSmartThingsGlossary(payload) {
  if (!payload || !payload.locale || !payload.version || !payload.checksum || !Array.isArray(payload.terms)) {
    throw new Error("locale, version, checksum, terms가 있는 glossary payload가 필요합니다.");
  }
  const terms = [...new Set(payload.terms.map((v) => String(v).trim()).filter(Boolean))];
  if (!terms.length) throw new Error("가져올 term이 없습니다.");
  return Excel.run(async (context) => {
    const sheets = context.workbook.worksheets;
    const existing = sheets.getItemOrNullObject("__ST_GLOSSARY");
    existing.load("isNullObject");
    await context.sync();
    if (!existing.isNullObject) throw new Error("__ST_GLOSSARY가 이미 있습니다. 기존 glossary를 덮어쓰지 않습니다.");
    const sheet = sheets.add("__ST_GLOSSARY");
    const rows = [["term", "locale", "version", "checksum"], ...terms.map((term) => [term, payload.locale, payload.version, payload.checksum])];
    const range = sheet.getRangeByIndexes(0, 0, rows.length, 4);
    range.values = rows;
    const table = sheet.tables.add(range, true, "ST_GLOSSARY");
    table.style = "TableStyleMedium2";
    sheet.visibility = Excel.SheetVisibility.hidden;
    sheet.protection.protect();
    await context.sync();
    return { imported: terms.length, locale: payload.locale, version: payload.version, checksum: payload.checksum };
  });
}

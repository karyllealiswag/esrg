/*
 * render_docx.js — Render the Chapter 4 specification (JSON) into a Word document.
 *
 * Purpose : Produce Chapter 4 in the thesis format (US Letter, 1-inch margins,
 *           Times New Roman 12 pt, 1.5 line spacing, APA 7 tables and figures,
 *           numbered OMML equations followed by a "where:" list).
 * Function : node render_docx.js <spec.json> <out.docx>. Each block of the spec is
 *           one of: h1 | h2 | h3 | p | eq | where | table | figure | pagebreak.
 *           Inline markup in text: *italic*, **bold**, _{subscript}, ^{superscript}.
 *           Math nodes: "text" | [nodes] | {f:[num,den]} | {sub:[b,s]} | {sup:[b,s]} |
 *           {subsup:[b,s,p]} | {sqrt:x} | {sum:{lo,hi,body}} | {rb:x} | {sb:x}.
 * Notes   : Requires the npm package "docx" (v9); see package.json.
 */
const fs = require("fs");
const path = require("path");
const d = require("docx");

const FONT = "Times New Roman";
const TEXT_W = 9360; // 6.5 in between 1-in margins, in DXA
const NONE = { style: d.BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = { style: d.BorderStyle.SINGLE, size: 6, color: "000000" };

// ── inline markup → TextRuns ────────────────────────────────────────────────
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})/g;
  let last = 0, m;
  const push = (t, o) => { if (t) out.push(new d.TextRun({ text: t, font: FONT, size: base.size || 24, ...base, ...o })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("*")) push(tok.slice(1, -1), { italics: true });
    else if (tok.startsWith("_{")) push(tok.slice(2, -1), { subScript: true });
    else push(tok.slice(2, -1), { superScript: true });
    last = m.index + tok.length;
  }
  push(text.slice(last), {});
  return out;
}

// ── math ────────────────────────────────────────────────────────────────────
function math(node) {
  if (node === null || node === undefined) return [];
  if (typeof node === "string") return [new d.MathRun(node)];
  if (Array.isArray(node)) return node.flatMap(math);
  if (node.f) return [new d.MathFraction({ numerator: math(node.f[0]), denominator: math(node.f[1]) })];
  if (node.sub) return [new d.MathSubScript({ children: math(node.sub[0]), subScript: math(node.sub[1]) })];
  if (node.sup) return [new d.MathSuperScript({ children: math(node.sup[0]), superScript: math(node.sup[1]) })];
  if (node.subsup) return [new d.MathSubSuperScript({ children: math(node.subsup[0]), subScript: math(node.subsup[1]), superScript: math(node.subsup[2]) })];
  if (node.sqrt !== undefined) return [new d.MathRadical({ children: math(node.sqrt) })];
  if (node.sum) {
    const o = { children: math(node.sum.body), subScript: math(node.sum.lo) };
    if (node.sum.hi) o.superScript = math(node.sum.hi);
    return [new d.MathSum(o)];
  }
  if (node.rb !== undefined) return [new d.MathRoundBrackets({ children: math(node.rb) })];
  if (node.sb !== undefined) return [new d.MathSquareBrackets({ children: math(node.sb) })];
  throw new Error("unknown math node " + JSON.stringify(node));
}

// ── block builders ──────────────────────────────────────────────────────────
const cellBorders = (top, bottom) => ({ top: top ? RULE : NONE, bottom: bottom ? RULE : NONE, left: NONE, right: NONE });

function heading(level, text) {
  const style = { h1: d.HeadingLevel.HEADING_1, h2: d.HeadingLevel.HEADING_2, h3: d.HeadingLevel.HEADING_3 }[level];
  return new d.Paragraph({
    heading: style, keepNext: true,
    alignment: level === "h1" ? d.AlignmentType.CENTER : d.AlignmentType.LEFT,
    spacing: { before: level === "h1" ? 0 : 240, after: 0, line: 360 },
    indent: level === "h3" ? { left: 720 } : undefined,
    children: runs(text, { bold: true, size: level === "h1" ? 28 : 24, color: "000000" }),
  });
}

function para(b) {
  return new d.Paragraph({
    alignment: d.AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 0, line: 360 },
    indent: b.noindent ? undefined : { firstLine: 720 },
    keepNext: !!b.keepNext,
    children: runs(b.text),
  });
}

function equation(b) {
  const cell = (children, w, align) => new d.TableCell({
    width: { size: w, type: d.WidthType.DXA }, borders: cellBorders(false, false),
    verticalAlign: d.VerticalAlign.CENTER, margins: { top: 60, bottom: 60, left: 0, right: 0 },
    children: [new d.Paragraph({ alignment: align, keepNext: true, children })],
  });
  return new d.Table({
    width: { size: TEXT_W, type: d.WidthType.DXA }, columnWidths: [1080, 7200, 1080],
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new d.TableRow({ cantSplit: true, children: [
      cell([], 1080, d.AlignmentType.LEFT),
      cell([new d.Math({ children: math(b.math) })], 7200, d.AlignmentType.CENTER),
      cell(runs(`(${b.num})`), 1080, d.AlignmentType.RIGHT),
    ] })],
  });
}

function where(b) {
  const out = [new d.Paragraph({ keepNext: true, spacing: { after: 0, line: 360 }, indent: { left: 720 }, children: runs("where:") })];
  b.items.forEach((it, i) => {
    const last = i === b.items.length - 1;
    out.push(new d.Paragraph({
      keepNext: !last, spacing: { after: last ? 200 : 0, line: 360 },
      indent: { left: 1440, hanging: 720 }, children: runs(it),
    }));
  });
  return out;
}

function tableBlock(b) {
  const out = [
    new d.Paragraph({ keepNext: true, spacing: { before: 240, after: 0, line: 360 }, children: runs(`Table ${b.num}`, { bold: true }) }),
    new d.Paragraph({ keepNext: true, spacing: { after: 0, line: 360 }, children: runs(b.title, { italics: true }) }),
  ];
  const ncol = b.widths.length;
  const total = b.widths.reduce((a, c) => a + c, 0);
  const widths = b.widths.map((w) => Math.round((w * TEXT_W) / total));
  widths[ncol - 1] += TEXT_W - widths.reduce((a, c) => a + c, 0);
  const align = (j) => ((b.align || [])[j] === "l" || (!b.align && j === 0) ? d.AlignmentType.LEFT : d.AlignmentType.CENTER);
  const mkCell = (c, j, opts) => {
    const o = typeof c === "object" && c !== null ? c : { text: String(c ?? "") };
    const span = o.span || 1;
    const w = widths.slice(j, j + span).reduce((a, x) => a + x, 0);
    return new d.TableCell({
      width: { size: w, type: d.WidthType.DXA }, columnSpan: span > 1 ? span : undefined,
      borders: cellBorders(opts.top, opts.bottom), verticalAlign: d.VerticalAlign.CENTER,
      margins: { top: 30, bottom: 30, left: 80, right: 80 },
      children: [new d.Paragraph({
        alignment: o.align === "l" ? d.AlignmentType.LEFT : o.align === "c" ? d.AlignmentType.CENTER : align(j),
        spacing: { line: 240 }, indent: o.indent ? { left: 200 } : undefined, keepNext: !!opts.keep,
        children: runs(o.text, { size: 20, bold: !!(opts.bold || o.bold), italics: !!o.italic }),
      })],
    });
  };
  const rows = [];
  const hdrs = b.header.length && Array.isArray(b.header[0]) ? b.header : [b.header];
  hdrs.forEach((h, hi) => {
    let j = 0;
    const cells = h.map((c) => { const cell = mkCell(c, j, { top: hi === 0, bottom: hi === hdrs.length - 1 || (typeof c === "object" && c && c.rule), bold: true, keep: true }); j += (c && c.span) || 1; return cell; });
    rows.push(new d.TableRow({ tableHeader: true, cantSplit: true, children: cells }));
  });
  b.rows.forEach((r, ri) => {
    const lastRow = ri === b.rows.length - 1;
    if (r.group !== undefined) {
      rows.push(new d.TableRow({ cantSplit: true, children: [mkCell({ text: r.group, span: ncol, italic: true, align: "l" }, 0, { bottom: lastRow, keep: !lastRow })] }));
      return;
    }
    let j = 0;
    // keep-with-next on every row but the last keeps a table on one page with its title when it fits
    const cells = r.map((c) => { const cell = mkCell(c, j, { bottom: lastRow, keep: !lastRow }); j += (c && c.span) || 1; return cell; });
    rows.push(new d.TableRow({ cantSplit: true, children: cells }));
  });
  out.push(new d.Table({
    width: { size: TEXT_W, type: d.WidthType.DXA }, columnWidths: widths,
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows,
  }));
  if (b.note) out.push(new d.Paragraph({ alignment: d.AlignmentType.JUSTIFIED, spacing: { before: 60, after: 240, line: 240 }, children: runs("*Note.* " + b.note, { size: 20 }) }));
  else out.push(new d.Paragraph({ spacing: { after: 240 }, children: [] }));
  return out;
}

function figure(b, baseDir) {
  const img = fs.readFileSync(path.resolve(baseDir, b.img));
  const w = 624; // 6.5 in at 96 dpi
  const h = Math.round((w * b.h) / b.w);
  const out = [
    new d.Paragraph({ keepNext: true, spacing: { before: 240, after: 0, line: 360 }, children: runs(`Figure ${b.num}`, { bold: true }) }),
    new d.Paragraph({ keepNext: true, spacing: { after: 0, line: 360 }, children: runs(b.title, { italics: true }) }),
    new d.Paragraph({ keepNext: true, alignment: d.AlignmentType.CENTER, children: [new d.ImageRun({ type: "png", data: img, transformation: { width: w, height: h } })] }),
  ];
  if (b.note) out.push(new d.Paragraph({ alignment: d.AlignmentType.JUSTIFIED, spacing: { before: 60, after: 240, line: 240 }, children: runs("*Note.* " + b.note, { size: 20 }) }));
  return out;
}

function reference(b) {
  return new d.Paragraph({ spacing: { after: 0, line: 360 }, indent: { left: 720, hanging: 720 }, children: runs(b.text) });
}

// ── main ────────────────────────────────────────────────────────────────────
const [specPath, outPath] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
const baseDir = path.dirname(path.resolve(specPath));
const children = [];
for (const b of spec.blocks) {
  if (["h1", "h2", "h3"].includes(b.t)) children.push(heading(b.t, b.text));
  else if (b.t === "p") children.push(para(b));
  else if (b.t === "eq") children.push(equation(b));
  else if (b.t === "where") children.push(...where(b));
  else if (b.t === "table") children.push(...tableBlock(b));
  else if (b.t === "figure") children.push(...figure(b, baseDir));
  else if (b.t === "ref") children.push(reference(b));
  else if (b.t === "pagebreak") children.push(new d.Paragraph({ children: [new d.PageBreak()] }));
  else throw new Error("unknown block " + b.t);
}
const doc = new d.Document({
  creator: spec.author || "",
  title: spec.title || "Chapter 4",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});
d.Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath); });

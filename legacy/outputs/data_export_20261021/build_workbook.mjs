import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(out, 'export_data.json'), 'utf8'));
const wb = Workbook.create();
const col = (n) => { let s = ''; for(n++; n; n = Math.floor((n-1)/26)) s = String.fromCharCode(65+(n-1)%26)+s; return s; };
for (const [index, def] of data.sheets.entries()) {
  const sh = wb.worksheets.add(def.name);
  const last = col(def.headers.length - 1), end = def.rows.length + 4;
  sh.showGridLines = false;
  sh.getRange(`A1:${last}${end}`).format.font = {name:'Arial', size:10, color:'#172B4D'};
  sh.getRange(`A1:${last}${end}`).format.rowHeight = 21;
  sh.getRange(`A1:${last}${end}`).format.verticalAlignment = 'center';
  sh.getRange(`A1:${last}${end}`).format.columnWidth = 17;
  sh.getRange('A1').values = [[`APIEMS ${def.name}`]];
  sh.getRange('A1').format.font = {name:'Arial',size:14,bold:true,color:'#172B4D'};
  sh.getRange('A1').format.rowHeight = 30;
  sh.getRange('A2').values = [[def.note]];
  sh.getRange('A2').format.font = {name:'Arial',size:10,color:'#516078'};
  sh.getRange('A3').values = [[`출처: ${def.source}`]];
  sh.getRange('A3').format.font = {name:'Arial',size:10,color:'#516078'};
  sh.getRange(`A4:${last}4`).values = [def.headers];
  sh.getRange(`A5:${last}${end}`).values = def.rows;
  if (def.name !== '안내') {
    const table = sh.tables.add(`A4:${last}${end}`, true, `ExportData${index}`);
    table.showFilterButton = true;
    sh.freezePanes.freezeRows(4);
    sh.getRange(`A5:${last}${end}`).setNumberFormat('#,##0');
    if (def.name === '학습표본') {
      sh.getRange(`B5:${last}${end}`).setNumberFormat('0.0000');
      for (const n of [1,2,3,6,7,12,13,23,28,30,31,32,33]) sh.getRange(`${col(n)}5:${col(n)}${end}`).setNumberFormat('#,##0');
    }
    for (const n of def.decimal) sh.getRange(`${col(n)}5:${col(n)}${end}`).setNumberFormat('#,##0.00');
    for (const n of def.percent) sh.getRange(`${col(n)}5:${col(n)}${end}`).setNumberFormat('0.0%');
  } else {
    sh.getRange(`A4:A${end}`).format.columnWidth = 24;
    sh.getRange(`B4:B${end}`).format.columnWidth = 112;
    sh.getRange(`B5:B${end}`).format.wrapText = true;
    sh.getRange(`A5:B${end}`).format.rowHeight = 34;
    sh.tabColor = '#243B53';
  }
  sh.getRange(`A4:${last}4`).format = {fill:'#243B53',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:def.name==='학습표본'?48:34};
  if (def.name === '기간별결과') sh.tabColor = '#4C6A92';
}
wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:'기간별결과',range:'A4:F8',maxChars:1400,tableMaxRows:5,tableMaxCols:6})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
for (const [i, def] of data.sheets.entries()) {
  const range = def.name==='안내'?'A1:B10':`A1:${col(Math.min(def.headers.length, 8)-1)}10`;
  const png = await wb.render({sheetName:def.name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(out, `preview_${i}.png`), new Uint8Array(await png.arrayBuffer()));
}
const result = await SpreadsheetFile.exportXlsx(wb);
const target = path.join(out,'APIEMS_데이터.xlsx');
await result.save(target);
console.log(JSON.stringify({output:target,counts:data.counts,profit:data.profit,verified:data.source_history_matched}));

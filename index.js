// 读取 HTML 文件内容
import fs from 'fs';
import path from 'path';

const htmlContent = fs.readFileSync(path.join(process.cwd(), 'dashboard.html'), 'utf8');

export default {
  async fetch(request) {
    return new Response(htmlContent, {
      headers: { 
        'content-type': 'text/html',
        'cache-control': 'public, max-age=3600'
      }
    });
  }
};
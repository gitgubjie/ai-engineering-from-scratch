# AI Engineering from Scratch — 本地部署版

Fork 自 [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)

---

## 本地运行

```bash
# 构建网站（每次同步后需重新运行）
node site/build.js

# 启动本地预览
npx serve site
# 访问 http://localhost:3000
```

---

## 同步上游

### 手动同步

```bash
./sync-upstream.sh
```

这会自动：拉取上游 → 合并到 main → 构建网站 → 推送到 origin。

### 自动同步

GitHub Actions 已配置每天凌晨 3 点自动同步。

如需手动触发：在 GitHub 仓库页面 → Actions → Sync Upstream → Run workflow

---

## 目录结构

| 路径 | 说明 |
|---|---|
| `phases/` | 20 个学习阶段，430 节课程内容 |
| `projects/` | 实战项目代码 |
| `site/` | 网站源码，`build.js` 生成 `data.js` |
| `glossary/` | 术语表 |
| `sync-upstream.sh` | 同步脚本 |

---

## Vercel 部署

1. Fork 本仓库到你的 GitHub
2. 在 Vercel Import → 选择本仓库
3. 无需任何配置，默认即可

构建命令：`node site/build.js`，输出目录：`site/`
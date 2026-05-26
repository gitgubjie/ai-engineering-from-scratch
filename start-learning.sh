#!/bin/bash
# start-learning.sh — 启动学习环境（Mac 默认路径）

export PATH="/opt/homebrew/bin:$PATH"

REPO_DIR="$HOME/Claude code/ai-engineering-from-scratch"

echo "=============================================="
echo "  🎓 启动 AI Engineering 学习环境"
echo "=============================================="
echo ""
echo "📂 仓库位置: $REPO_DIR"
echo ""

# Check Python
if command -v python3 &>/dev/null; then
  echo "✅ Python: $(python3 --version)"
else
  echo "❌ Python 未安装"
fi

# Check Node
if command -v node &>/dev/null; then
  echo "✅ Node.js: $(node --version)"
else
  echo "❌ Node.js 未安装"
fi

# Check Rust
if command -v rustc &>/dev/null; then
  echo "✅ Rust: $(rustc --version)"
else
  echo "⚠️  Rust 未安装（可选）"
fi

echo ""
echo "📖 启动本地网站..."

cd "$REPO_DIR"
npx serve site -p 3000 &

sleep 2

echo ""
echo "  ✅ 网站已启动！"
echo ""
echo "  📖 课程主页:   http://localhost:3000"
echo "  📂 文件浏览:   http://localhost:3000/browse"
echo ""
echo "  开始学习吧！按 Ctrl+C 停止服务器"
echo ""
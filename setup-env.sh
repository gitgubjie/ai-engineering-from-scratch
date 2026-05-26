#!/bin/bash
# setup-env.sh — 检查并配置学习环境

export PATH="/opt/homebrew/bin:$PATH"

echo "=============================================="
echo "  🎓 AI Engineering — 环境检查"
echo "=============================================="
echo ""

# Python
echo "🐍 Python:"
if command -v python3 &>/dev/null; then
  echo "   $(python3 --version)"
else
  echo "   未找到，请安装: brew install python"
fi

# Node.js
echo ""
echo "📦 Node.js:"
if command -v node &>/dev/null; then
  echo "   $(node --version)"
else
  echo "   未找到，请安装 Node.js"
fi

# Rust
echo ""
echo "🦀 Rust:"
if command -v rustc &>/dev/null; then
  echo "   $(rustc --version)"
else
  echo "   未找到（可选，用于 Rust 课程）"
fi

# Docker
echo ""
echo "🐳 Docker:"
if command -v docker &>/dev/null; then
  echo "   $(docker --version)"
else
  echo "   未找到（Phase 0 需要）"
fi

# CUDA (Mac 通常没有，跳过)
echo ""
echo "🖥️  GPU:"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi | head -5
else
  echo "   Mac 无 NVIDIA GPU（正常）"
fi

echo ""
echo "=============================================="
echo "  🚀 启动课程网站"
echo "=============================================="
echo ""
echo "  cd ~/Claude\\ code/ai-engineering-from-scratch"
echo "  node server.js"
echo ""
echo "  然后打开: http://localhost:3000"
echo ""
echo "=============================================="
echo "  📚 开始学习"
echo "=============================================="
echo ""
echo "  Phase 0: 开发环境配置"
echo "  Phase 1: 数学基础"
echo "  Phase 2: 机器学习"
echo "  Phase 3: 深度学习"
echo "  Phase 4: 计算机视觉"
echo "  Phase 5: NLP"
echo "  ..."
echo ""
echo "  在 http://localhost:3000/browse 可浏览所有课程文件"
echo ""
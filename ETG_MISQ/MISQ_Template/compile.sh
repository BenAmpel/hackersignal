#!/bin/bash

# MISQ Template Compilation Script
# This script compiles the LaTeX document using XeLaTeX and Biber
# Usage: ./compile.sh [filename]
# If no filename is provided, defaults to misq_template.tex

# Set the filename (default to misq_template.tex if not provided)
FILENAME="${1:-misq_template.tex}"

# Check if the file exists
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' not found."
    exit 1
fi

# Get the base name without extension
BASENAME="${FILENAME%.tex}"

echo "Compiling $FILENAME..."
echo ""

# First pass: XeLaTeX
echo "Running XeLaTeX (pass 1/3)..."
xelatex -interaction=nonstopmode "$FILENAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: XeLaTeX compilation failed. Check ${BASENAME}.log for details."
    exit 1
fi

# Run Biber for bibliography
echo "Running Biber..."
biber "$BASENAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Warning: Biber may have encountered issues. Check ${BASENAME}.blg for details."
fi

# Second pass: XeLaTeX
echo "Running XeLaTeX (pass 2/3)..."
xelatex -interaction=nonstopmode "$FILENAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: XeLaTeX compilation failed. Check ${BASENAME}.log for details."
    exit 1
fi

# Third pass: XeLaTeX (for cross-references)
echo "Running XeLaTeX (pass 3/3)..."
xelatex -interaction=nonstopmode "$FILENAME" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: XeLaTeX compilation failed. Check ${BASENAME}.log for details."
    exit 1
fi

echo ""
echo "Compilation complete! Output: ${BASENAME}.pdf"
echo ""
echo "To see detailed output, remove '> /dev/null 2>&1' from the script or run:"
echo "  xelatex $FILENAME"
echo "  biber $BASENAME"
echo "  xelatex $FILENAME"
echo "  xelatex $FILENAME"


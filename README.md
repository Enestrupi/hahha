# 📚 AI Ebook Maker - Real Book Photos to Clean Ebook

Turn 316 real photos of a physical book (on wooden table with shadows) into a clean ebook PDF like a digital textbook.

> Built for MATEMATIKA II textbook - removes background, fixes perspective, makes white pages.

## ✨ Features

- **📸 Background Removal**: Removes wooden table, shadows, extra captured areas
- **📖 Perspective Fix**: Straightens curved pages, auto-detects page corners
- **🤖 AI Cleaning**: Shadow removal, white background, text sharpening
- **📦 Bulk Upload**: Handles 316 photos at once (chunked processing, never crashes)
- **🗜️ ZIP Support**: Upload ZIP with 316 photos inside
- **📕 PDF Export**: Clean A4 ebook with page numbers

## 🎯 Before vs After

**Before:** Photo of real book on wooden table, shadows, curved pages, background visible
**After:** Clean white page, no background, straight, like digital ebook example

## 🚀 Quick Start

### Web App (No Install)

1. Open `index-bulk-316.html` for 316 photos at once
2. Or `index-v3-scanner.html` for real book photos with background removal
3. Or `index.html` for digital screenshots
4. Drop photos → AI scans → Export PDF

### Python Batch (Best for 316+ photos)

```bash
pip install opencv-python pillow flask flask-cors
python scanner_server.py  # Starts AI scanner API on port 8001
# Then open HTML file

# Or batch process directly:
python batch_real_book.py --input ./my_316_photos --output MATEMATIKA_II_clean.pdf
```

## 📂 Files

- `index-bulk-316.html` - **BEST for 316 photos** - Chunked processing, ZIP support, never crashes
- `index-v3-scanner.html` - Real book photos → Clean ebook (background removal)
- `index-v2.html` - Handles 16MB+ tall screenshots
- `index.html` - Basic version
- `scanner_server.py` - AI backend (OpenCV) - removes background, fixes perspective
- `batch_real_book.py` - Batch process real book photos to PDF
- `batch_processor.py` - Batch process digital screenshots

## 📱 How to Upload 316 Photos at Once

### Method 1: ZIP (Recommended)
1. iPhone: Photos → Select 316 photos → Share → Save to Files
2. Files app → Select all → Compress (creates ZIP)
3. Upload ZIP to app

### Method 2: Folder Upload
Put all 316 in folder → Click "Select Folder" button

### Method 3: Select All at Once
File picker → Select 316 → App processes in chunks of 20

## 🛠️ Tech Stack

- Frontend: Vanilla JS, Canvas, PDF-Lib, JSZip
- Backend: Python, OpenCV, Flask
- AI: Document detection, perspective transform, shadow removal via illumination correction

## 📄 License

MIT - Free for educational use

## 🙏 Credits

Built for MATEMATIKA II textbook scanning - turns physical book photos into clean digital ebook like the example with blue sky cover.

## 🌐 Live Demo

After pushing to GitHub, enable GitHub Pages in Settings → Pages → Deploy from branch: main
Then your app will be live at: `https://Enestrupi.github.io/hahha/index-bulk-316.html`

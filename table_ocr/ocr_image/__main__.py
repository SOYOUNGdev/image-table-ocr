import argparse
import pytesseract
import sys
import cv2

description = """Takes an image and performs OCR.
If mode is 'table', it extracts table data into a CSV file.
If mode is 'text', it extracts general text into a TXT file.
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("image", help="Filepath of the image to perform OCR")
parser.add_argument("mode", choices=["table", "text"], help="OCR mode (table or text)")

args, tess_args = parser.parse_known_args()

def ocr_table(image):
    """
    테이블 OCR 실행 (CSV용)
    """
    text = pytesseract.image_to_string(image, lang="table-ocr" config="--psm 6")
    return text.strip()

def ocr_paragraph(image):
    """
    일반 텍스트 OCR 실행 (TXT 저장용)
    """
    text = pytesseract.image_to_string(image, lang="eng+kor", config="--psm 3")
    return text.strip()

def main(image_path, mode):
    """
    이미지 파일을 입력받아 OCR을 실행.
    mode="table"이면 표 OCR, mode="text"이면 일반 텍스트 OCR.
    """
    image = cv2.imread(image_path)

    if image is None:
        print("❌ 오류: 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)

    if mode == "table":
        result = ocr_table(image)
        output_file = "output_table.csv"
    else:
        result = ocr_paragraph(image)
        output_file = "output_text.txt"

    # 파일 저장
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"✅ OCR 결과가 {output_file} 파일에 저장되었습니다!")

if __name__ == "__main__":
    main(args.image, args.mode)
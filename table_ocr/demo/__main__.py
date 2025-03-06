import os
import sys
import requests
import cv2
import numpy as np
import pytesseract
import table_ocr.util
import table_ocr.extract_tables
import table_ocr.extract_cells
import table_ocr.ocr_image
import table_ocr.ocr_to_csv

def download_image_to_tempdir(url, filename=None):
    """
    이미지 URL을 다운로드하여 임시 폴더에 저장.
    """
    if filename is None:
        filename = os.path.basename(url)
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"❌ 오류: 이미지 다운로드 실패 (HTTP {response.status_code})")
        sys.exit(1)

    tempdir = table_ocr.util.make_tempdir("demo")
    filepath = os.path.join(tempdir, filename)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content():
            f.write(chunk)
    
    print(f"📂 이미지가 다운로드됨: {filepath}")  # ✅ 디버깅용 출력 추가
    return filepath

def load_image(image_path):
    """
    이미지 파일을 로드하고, 문제가 있으면 오류 출력.
    """
    if not os.path.exists(image_path):
        print(f"❌ 오류: 파일이 존재하지 않습니다. ({image_path})")
        sys.exit(1)

    # ✅ 이미지 로드 시 다양한 방법 테스트
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"⚠️ 경고: IMREAD_COLOR 로드 실패. 다른 모드 시도 중...")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"❌ 오류: 이미지 로드 실패 (경로: {image_path})")
        sys.exit(1)

    print(f"✅ 이미지 로드 성공: {image.shape}")  # (높이, 너비, 채널 수)
    return image

def detect_table_regions(image):
    """
    이미지에서 표 영역을 감지하여 좌표를 반환.
    """
    if image is None:
        print("❌ 오류: detect_table_regions()에서 이미지가 None입니다.")
        sys.exit(1)

    # ✅ 디버깅 정보 추가
    print(f"🔍 detect_table_regions() 실행: type(image)={type(image)}")

    # ✅ 이미지가 numpy 배열이 아닌 경우 변환
    if not isinstance(image, np.ndarray):
        print("⚠️ 이미지 형식이 numpy 배열이 아님. 변환 시도 중...")
        image = np.array(image)

    print(f"✅ 변환 후 type(image)={type(image)}, shape={image.shape}")

    # ✅ 이미지가 정상적으로 변환되었는지 확인
    if not isinstance(image, np.ndarray):
        print("❌ 오류: 이미지 변환 실패. numpy 배열이 아닙니다.")
        sys.exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_mask = cv2.add(horizontal_lines, vertical_lines)

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        tables.append((x, y, w, h))

    return tables


def remove_table_from_image(image, tables):
    """
    감지된 표 영역을 마스킹하여 일반 텍스트만 남김.
    """
    mask = np.ones_like(image) * 255  # 흰색 마스크 생성
    for (x, y, w, h) in tables:
        mask[y:y+h, x:x+w] = 0  # 표 부분을 검정색(0)으로 덮음

    text_only_image = cv2.bitwise_and(image, mask)  # 표를 제외한 텍스트만 남김
    return text_only_image

def extract_text(image, output_txt="output_text.txt"):
    """
    표를 제외한 일반 텍스트를 OCR 후 TXT 파일로 저장.
    텍스트가 없는 경우 저장하지 않음.
    """
    # 🔥 OCR 실행
    text_result = pytesseract.image_to_string(image, lang="eng+kor", config="--psm 3")

    # ✅ OCR 결과에서 불필요한 줄바꿈 제거 및 공백 검사
    cleaned_text = "\n".join([line.strip() for line in text_result.split("\n") if line.strip()])

    # ✅ 텍스트가 아예 없는 경우 저장하지 않음
    if not cleaned_text.strip():
        print("⚠️ 일반 텍스트가 감지되지 않았습니다. `output_text.txt`를 생성하지 않습니다.")
        return

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ 일반 텍스트 OCR 결과가 {output_txt} 파일에 저장되었습니다!")

def process_image(url, output_csv="output_table.csv", output_txt="output_text.txt"):
    """
    테이블과 일반 텍스트를 OCR하고 결과를 저장.
    """
    image_filepath = download_image_to_tempdir(url)
    
    # ✅ 이미지 로드 및 검증
    image = load_image(image_filepath)  # ✅ image 변수를 실제 이미지로 유지

    # ✅ 1. 표 OCR 실행
    image_tables = table_ocr.extract_tables.main([image_filepath])

    tables_detected = False  # 표가 감지되었는지 여부
    for _, tables in image_tables:
        if tables:
            tables_detected = True
            print(f"📌 표 감지됨 → CSV 저장 중...")
            for table in tables:
                cells = table_ocr.extract_cells.main(table)
                ocr = [
                    table_ocr.ocr_image.main(cell, None)
                    for cell in cells
                ]
                csv_data = table_ocr.ocr_to_csv.text_files_to_csv(ocr)

                # ✅ CSV 저장
                with open(output_csv, "w", encoding="utf-8") as f:
                    f.write(csv_data)

                print(f"✅ 표 OCR 결과가 {output_csv} 파일에 저장되었습니다!")

    # ✅ 2. 표가 감지된 경우, 일반 텍스트 OCR 실행
    if tables_detected:
        print("📌 표 영역을 제외하고 일반 텍스트 OCR 수행 중...")
        tables = detect_table_regions(image)  # ✅ `image_filepath`가 아니라 `image`를 전달
        text_only_image = remove_table_from_image(image, tables)

        if text_only_image is None:
            print("⚠️ 텍스트만 남긴 이미지가 None입니다. 일반 텍스트 OCR을 건너뜁니다.")
        else:
            extract_text(text_only_image, output_txt)
    else:
        print("⚠️ 표가 감지되지 않았으므로, 일반 텍스트만 OCR 실행")
        extract_text(image, output_txt)

    return output_csv, output_txt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 사용법: python -m table_ocr.demo <이미지 URL>")
        sys.exit(1)

    csv_output, txt_output = process_image(sys.argv[1], "output_table.csv", "output_text.txt")

    print("\nHere is the extracted OCR data:")
    print(f"📄 CSV File: {csv_output}")
    print(f"📄 TXT File: {txt_output}")
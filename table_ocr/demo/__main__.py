import os
import sys
import requests
import table_ocr.util
import table_ocr.extract_tables
import table_ocr.extract_cells
import table_ocr.ocr_image
import table_ocr.ocr_to_csv

def download_image_to_tempdir(url, filename=None):
    if filename is None:
        filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    tempdir = table_ocr.util.make_tempdir("demo")
    filepath = os.path.join(tempdir, filename)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content():
            f.write(chunk)
    return filepath

def main(url, output_csv="output.csv"):
    image_filepath = download_image_to_tempdir(url)
    image_tables = table_ocr.extract_tables.main([image_filepath])
    print("Running `{}`".format(f"extract_tables.main([{image_filepath}])."))
    print("Extracted the following tables from the image:")
    print(image_tables)

    for image, tables in image_tables:
        print(f"Processing tables for {image}.")
        for table in tables:
            print(f"Processing table {table}.")
            cells = table_ocr.extract_cells.main(table)
            ocr = [
                table_ocr.ocr_image.main(cell, None)
                for cell in cells
            ]
            print("Extracted {} cells from {}".format(len(ocr), table))
            print("Cells:")

            for c, o in zip(cells[:3], ocr[:3]):
                with open(o) as ocr_file:
                    text = ocr_file.read().strip()
                    print("{}: {}".format(c, text))

            if len(cells) > 3:
                print("...")

            # CSV 데이터 저장 (기존 기능 활용)
            csv_data = table_ocr.ocr_to_csv.text_files_to_csv(ocr)

            # CSV 파일 저장 추가
            with open(output_csv, "w", encoding="utf-8") as f:
                f.write(csv_data)

            print(f"✅ OCR 결과가 {output_csv} 파일에 저장되었습니다!")
            return csv_data  # CSV 데이터 반환

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 사용법: python -m table_ocr.demo <이미지 URL>")
        sys.exit(1)

    csv_output = main(sys.argv[1], "output.csv")

    print()
    print("Here is the entire CSV output:")
    print()
    print(csv_output)
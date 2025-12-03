import dart_fss as dart
import pandas as pd
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
import os
import time

# 한글 폰트 적용
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False

# Open DART API Key
api_key = 'ff5d4bff214441585f355c4297775e3fdb09259b'
dart.set_api_key(api_key)


def get_kospi_list():
    file_path = 'data/상장법인목록.csv'

    df = pd.read_csv(file_path, encoding='cp949')  # KRX CSV는 대부분 cp949

    # 상장일을 datetime으로 변환
    df['상장일'] = pd.to_datetime(df['상장일'], errors='coerce')

    # 필터링
    kospi_list = df[
        (df['시장구분'] == '유가') &
        (df['상장일'] <= '2014-01-01') &
        (df['결산월'] == '12월')
    ]

    # 결과 확인
    print(f"조건에 맞는 기업 수: {len(kospi_list)}개")
    print(kospi_list)
    
    return kospi_list


def match_corpcode_krx_to_dart(kospi_list):
    # XML 파일 불러오기
    tree = ET.parse('data/dart_corp_list.xml')
    root = tree.getroot()

    # 리스트로 변환
    corp_list = []
    for lst in root.findall('list'):
        corp_list.append({
            'corp_code': lst.find('corp_code').text,
            'corp_name': lst.find('corp_name').text,
            'stock_code': lst.find('stock_code').text,
            'modify_date': lst.find('modify_date').text
        })

    dart_df = pd.DataFrame(corp_list)

    # CSV에서 필터링한 회사명과 매칭
    kospi_df = kospi_list.merge(dart_df[['corp_name', 'corp_code']],
                                left_on='회사명', right_on='corp_name', how='left')
    kospi_df = kospi_df.rename(columns={'회사명':'corp_name'})

    # corp_code를 문자열로 강제 변환, 8자리로 채우기
    kospi_df['corp_code'] = kospi_df['corp_code'].astype(str).str.zfill(8)
    # Excel에서도 안전하게 문자열로 인식되도록 따옴표 추가
    kospi_df['corp_code'] = kospi_df['corp_code'].apply(lambda x: f'"{x}"')

    kospi_df.to_csv('./data/kospi_corps.csv', encoding="utf-8-sig", index=False)
    print('kospi_corps.csv 저장 완료')
    return kospi_df


def get_financial_raw_data():
    kospi_df = pd.read_csv('./data/kospi_corps.csv', encoding='utf-8-sig')
    kospi_count = len(kospi_df)
    result = []  # 결과 저장 리스트
    
    for row in kospi_df.itertuples():
        corp_name = row.corp_name
        corp_code = row.corp_code.replace('"','')
        print(f"\n{corp_name} 데이터 수집 중")

        for year in range(2022, 2024):
            for reprt_code, reprt_name in [('11012', '반기보고서'), ('11011', '사업보고서')]:

                try:
                    url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
                    params = {
                        'crtfc_key': api_key,
                        'corp_code': corp_code,
                        'bsns_year': year,
                        'reprt_code': reprt_code
                    }

                    resp = requests.get(url, params=params)
                    js = resp.json()
                    print(js['status'])
                    if js['status'] == '000':
                        financial_list = js['list']
                        df = pd.DataFrame(financial_list)
                        
                        cfs_df = df[df['fs_div'] == 'CFS']
                        ofs_df = df[df['fs_div'] == 'OFS']

                        if not cfs_df.empty:
                            df = cfs_df  # CFS가 있으면 CFS 사용
                        elif not ofs_df.empty:
                            df = ofs_df  # 없으면 OFS 사용
                        else:
                            print("CFS, OFS 모두 데이터 없음")
                            continue
                        
                        if reprt_name == "반기보고서":
                            sales = df[df["account_nm"] == "매출액"]["thstrm_add_amount"]
                            op_profit = df[df["account_nm"].isin(["영업이익", "영업손익"])]["thstrm_add_amount"]
                        else:
                            sales = df[df["account_nm"] == "매출액"]["thstrm_amount"]
                            op_profit = df[df["account_nm"].isin(["영업이익", "영업손익"])]["thstrm_amount"]
                            
                        result.append({
                            "기업명": corp_name,
                            "연도": year,
                            "보고서유형": reprt_name,
                            "매출액": sales.iloc[0] if not sales.empty else None,
                            "영업이익": op_profit.iloc[0] if not op_profit.empty else None
                        })
                    else:
                        print(f"데이터 없음: {js.get('message')}")

                    print(f"{corp_name} {year}년 {reprt_name} 수집 완료")
                    time.sleep(0.3)

                except Exception as e:
                    print(f"{year}년 {reprt_name} 실패: {e}")
                    
        print(f"진행 상황 : {len(result)} / {kospi_count}")
        
    # 결과 합치기 및 CSV 저장
    os.makedirs('./data', exist_ok=True)
    if result:
        final_df = pd.DataFrame(result)
    else:
        final_df = pd.DataFrame()
        print("Error : 값이 존재하지 않습니다.")
    final_df.to_csv('./data/financial_raw.csv', encoding="utf-8-sig", index=False)
    print('financial_raw.csv 저장 완료')
    return final_df

if __name__ == '__main__':
    kospi_list = get_kospi_list()
    kospi_df = match_corpcode_krx_to_dart(kospi_list)
    financial_raw_df = get_financial_raw_data()
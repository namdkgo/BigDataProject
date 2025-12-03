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

def corpcode_download():
    try:
        url = f"https://opendart.fss.or.kr/api/corpCode.xml"
        params = {
            'crtfc_key': api_key
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()

        # XML 파일로 저장
        with open("corpCode.xml", "wb") as f:
            f.write(resp.content)
        print("XML 파일 다운로드 완료!")

    except Exception as e:
        print("실패:", e)


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


def get_emp_raw_data():
    kospi_df = pd.read_csv('./data/kospi_corps.csv', encoding='utf-8-sig')
    kospi_count = len(kospi_df)
    result = []  # 결과 저장 리스트
    
    needed_cols = [ # DART 직원현황에서 가져올 컬럼 정의
        'corp_code',
        'corp_name',
        'sexdstn',
        'fo_bbm',
        'reform_bfe_emp_co_rgllbr',
        'reform_bfe_emp_co_cnttk',
        'sm',
        'fyer_salary_totamt',
    ]
    
    for row in kospi_df.itertuples():
        corp_name = row.corp_name
        corp_code = row.corp_code.replace('"','')
        print(f"\n{corp_name} 데이터 수집 중")

        for year in range(2022, 2024):
            for reprt_code, reprt_name in [('11012', '반기보고서'), ('11011', '사업보고서')]:

                try:
                    url = "https://opendart.fss.or.kr/api/empSttus.json"
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
                        emp_list = js['list']
                        emp_df = pd.DataFrame(emp_list)
                        
                        emp_df = emp_df[[col for col in needed_cols if col in emp_df.columns]]
                        # 결과 누적
                        emp_df['연도'] = year
                        emp_df['보고서유형'] = reprt_name
                        
                        emp_df = emp_df.rename(columns={
                            'corp_code' : '종목코드',
                            'corp_name' : '기업명',
                            'sexdstn' : '성별',
                            'fo_bbm' : '사업부문',
                            'reform_bfe_emp_co_rgllbr' : '정규직',
                            'reform_bfe_emp_co_cnttk' : '계약직',
                            'sm' : '직원합',
                            'fyer_salary_totamt' : '총급여'
                        })
                        
                        result.append(emp_df)
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
        final_df = pd.concat(result, ignore_index=True)
    else:
        final_df = pd.DataFrame()
    final_df.to_csv('./data/emp_raw.csv', encoding="utf-8-sig", index=False)
    print('emp_raw.csv 저장 완료')
    return final_df


def purify_emp_data():
    emp_raw_df = pd.read_csv('./data/emp_raw.csv')
    # 숫자 타입 변환 (혹시 문자열로 되어있을 경우 대비)
    for col in ['정규직', '계약직', '직원합', '총급여']:
        emp_raw_df[col] = emp_raw_df[col].astype(str).str.replace(',', '')
        emp_raw_df[col] = pd.to_numeric(emp_raw_df[col], errors='coerce').fillna(0)

    # 그룹화 및 합산
    emp_df = emp_raw_df.groupby(
        ['기업명', '연도', '보고서유형', '성별'],
        as_index=False
    ).agg({
        '정규직': 'sum',
        '계약직': 'sum',
        '총급여': 'sum'
    })
    
    emp_df.to_csv('./data/emp.csv', encoding="utf-8-sig", index=False)
    print('emp.csv 저장 완료')
    return emp_df
    
def final_emp_data():
    emp_df = pd.read_csv('./data/emp.csv')
    
    # 2014~2024년, 반기보고서와 사업보고서가 모두 있는 회사+성별만 남김.
    years = list(range(2014, 2025))
    report_types = ['반기보고서', '사업보고서']

    # 각 회사+성별 그룹별 (연도, 보고서유형) 개수 확인
    grouped = emp_df.groupby(['기업명', '성별'])
    
    valid_idx = []
    for (corp, gender), group in grouped:
        # (연도, 보고서유형) 조합 추출
        pairs = set([tuple(x) for x in group[['연도', '보고서유형']].values])
        # 모든 연도·보고서유형이 존재하면 valid
        all_pairs = set((y, r) for y in years for r in report_types)
        if all_pairs.issubset(pairs):
            valid_idx.append((corp, gender))
    
    # 필터링
    final_emp_df = emp_df.set_index(['기업명', '성별']).loc[valid_idx].reset_index()
    
    final_emp_df.to_csv('./data/emp_final.csv', encoding="utf-8-sig", index=False)
    print('emp_complete.csv 저장 완료')
    
    return final_emp_df

if __name__ == '__main__':
    # kospi_list = get_kospi_list()
    # kospi_df = match_corpcode_krx_to_dart(kospi_list)
    emp_raw_df = get_emp_raw_data()
    emp_df = purify_emp_data()
    # final_emp_data()
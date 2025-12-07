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
api_key = 'api_key_secret'
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
    result = []  # 결과 저장 리스트
    
    needed_cols = [ # DART 직원현황에서 가져올 컬럼 정의
        'corp_code',
        'corp_name',
        'sexdstn',
        'fo_bbm',
        'rgllbr_co',
        'cnttk_co',
        'sm',
        'fyer_salary_totamt',
    ]
    
    for row in kospi_df.itertuples():
        corp_name = row.corp_name
        corp_code = row.corp_code.replace('"','')
        print(f"\n{corp_name} 데이터 수집 중")

        for year in range(2016, 2025):
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
                            'rgllbr_co' : '정규직',
                            'cnttk_co' : '계약직',
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
                    
        print(f"진행 상황 : {len(result)} ")
        
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
    
    emp_df.to_csv('./data/emp_combine.csv', encoding="utf-8-sig", index=False)
    print('emp_combine.csv 저장 완료')
    return emp_df
    
def valid_emp_data():
    emp_df = pd.read_csv('./data/emp.csv')
    
    # 2016~2024년, 반기보고서와 사업보고서가 모두 있는 회사+성별만 남김.
    years = list(range(2016, 2025))
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
    valid_emp_df = emp_df.set_index(['기업명', '성별']).loc[valid_idx].reset_index()
    
    valid_emp_df.to_csv('./data/emp_valid.csv', encoding="utf-8-sig", index=False)
    print('emp_valid.csv 저장 완료')
    
    return valid_emp_df


def transform_emp_final():
    import pandas as pd
    df = pd.read_csv('./data/emp_valid.csv')

    # ---------------------------
    # 1. 숫자형 컬럼 변환
    # ---------------------------
    numeric_cols = ['정규직', '계약직', '총급여']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ---------------------------
    # 2. 반기/사업 분리
    # ---------------------------
    semi = df[df['보고서유형'] == '반기보고서']
    annual = df[df['보고서유형'] == '사업보고서']

    # ---------------------------
    # 3. 반기-사업 merge → 순수 연간 급여 계산
    # ---------------------------
    merged = pd.merge(
        annual,
        semi,
        on=['기업명', '연도', '성별'],
        suffixes=('_annual', '_semi'),
        how='left'
    )

    merged['순수연간급여'] = merged['총급여_annual'] - merged.get('총급여_semi', 0)
    merged.loc[merged['순수연간급여'] <= 0, '순수연간급여'] = merged['총급여_annual']

    # ---------------------------
    # 4. 사업보고서 pivot
    # ---------------------------
    pivot_annual = merged.pivot_table(
        index=['기업명', '연도'],
        columns='성별',
        values=['정규직_annual', '계약직_annual', '순수연간급여'],
        aggfunc='sum',
        fill_value=0
    )
    pivot_annual.columns = ['_'.join(col).strip() for col in pivot_annual.columns.values]
    pivot_annual = pivot_annual.reset_index()

    # ---------------------------
    # 5. 반기보고서 pivot
    # ---------------------------
    pivot_semi = semi.pivot_table(
        index=['기업명', '연도'],
        columns='성별',
        values=['정규직', '계약직', '총급여'],
        aggfunc='sum',
        fill_value=0
    )
    pivot_semi.columns = ['_'.join(col).strip() for col in pivot_semi.columns.values]
    pivot_semi = pivot_semi.reset_index()

    # ---------------------------
    # 6. 사업보고서 계산 테이블 생성
    # ---------------------------
    annual_rows = []
    for row in pivot_annual.itertuples():
        corp, year = row.기업명, row.연도

        m_reg = getattr(row, '정규직_annual_남', 0)
        m_con = getattr(row, '계약직_annual_남', 0)
        m_pay = getattr(row, '순수연간급여_남', 0)

        f_reg = getattr(row, '정규직_annual_여', 0)
        f_con = getattr(row, '계약직_annual_여', 0)
        f_pay = getattr(row, '순수연간급여_여', 0)

        tot_reg, tot_con = m_reg + f_reg, m_con + f_con
        tot_pay = m_pay + f_pay
        tot_emp = tot_reg + tot_con
        avg_pay = tot_pay / tot_emp if tot_emp > 0 else None

        annual_rows.append({
            '기업명': corp,
            '연도': year,
            '보고서유형': '사업보고서',
            '평균급여': avg_pay,
            '남성직원수': m_reg + m_con,
            '여성직원수': f_reg + f_con,
            '정규직수': tot_reg,
            '계약직수': tot_con,
            '남성정규직수': m_reg,
            '남성계약직수': m_con,
            '여성정규직수': f_reg,
            '여성계약직수': f_con,
        })

    annual_df = pd.DataFrame(annual_rows)

    # ---------------------------
    # 7. 반기보고서 테이블 생성
    # ---------------------------
    semi_rows = []
    for row in pivot_semi.itertuples():
        corp, year = row.기업명, row.연도

        m_reg = getattr(row, '정규직_남', 0)
        m_con = getattr(row, '계약직_남', 0)
        m_pay = getattr(row, '총급여_남', 0)

        f_reg = getattr(row, '정규직_여', 0)
        f_con = getattr(row, '계약직_여', 0)
        f_pay = getattr(row, '총급여_여', 0)

        tot_reg, tot_con = m_reg + f_reg, m_con + f_con
        tot_pay = m_pay + f_pay
        tot_emp = tot_reg + tot_con
        avg_pay = tot_pay / tot_emp if tot_emp > 0 else None

        semi_rows.append({
            '기업명': corp,
            '연도': year,
            '보고서유형': '반기보고서',
            '평균급여': avg_pay,
            '남성직원수': m_reg + m_con,
            '여성직원수': f_reg + f_con,
            '정규직수': tot_reg,
            '계약직수': tot_con,
            '남성정규직수': m_reg,
            '남성계약직수': m_con,
            '여성정규직수': f_reg,
            '여성계약직수': f_con,
        })

    semi_df = pd.DataFrame(semi_rows)

    # ---------------------------
    # 8. 반기 + 사업 통합
    # ---------------------------
    total = pd.concat([semi_df, annual_df])
    total = total.sort_values(['기업명', '연도', '보고서유형']).reset_index(drop=True)

    # ---------------------------
    # 9. 증감율 계산
    # ---------------------------
    rate_cols = [
        '평균급여', '남성직원수', '여성직원수',
        '정규직수', '계약직수',
        '남성정규직수', '남성계약직수',
        '여성정규직수', '여성계약직수'
    ]

    for col in rate_cols:
        total[col + '증감율'] = None

    for corp in total['기업명'].unique():
        corp_df = total[total['기업명'] == corp]

        for year in corp_df['연도'].unique():
            ydf = corp_df[corp_df['연도'] == year]

            # (1) 반기 = 전년도 사업보고서 대비
            h1 = ydf[ydf['보고서유형'] == '반기보고서']
            prev = total[
                (total['기업명'] == corp) &
                (total['연도'] == year - 1) &
                (total['보고서유형'] == '사업보고서')
            ]

            if not h1.empty and not prev.empty:
                h1_idx = h1.index[0]
                for col in rate_cols:
                    before = prev[col].values[0]
                    after = h1[col].values[0]
                    if pd.notna(before) and before != 0:
                        total.loc[h1_idx, col + '증감율'] = (after - before) / before * 100

            # (2) 사업 = 같은 연도 반기보고서 대비
            ann = ydf[ydf['보고서유형'] == '사업보고서']
            if not ann.empty and not h1.empty:
                ann_idx = ann.index[0]
                for col in rate_cols:
                    before = h1[col].values[0]
                    after = ann[col].values[0]
                    if pd.notna(before) and before != 0:
                        total.loc[ann_idx, col + '증감율'] = (after - before) / before * 100

    # ---------------------------
    # 10. 저장
    # ---------------------------
    total.to_csv('./data/emp_final.csv', index=False, encoding='utf-8-sig')
    print("emp_final.csv 저장 완료!")

    return total


def get_financial_raw_data():
    kospi_df = pd.read_csv('./data/kospi_corps.csv', encoding='utf-8-sig')
    kospi_count = len(kospi_df)
    result = []  # 결과 저장 리스트
    
    for row in kospi_df.itertuples():
        corp_name = row.corp_name
        corp_code = row.corp_code.replace('"','')
        print(f"\n{corp_name} 데이터 수집 중")

        for year in range(2016, 2025):
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


def purify_financial():
    # 1. raw 파일 불러오기
    df = pd.read_csv('./data/financial_raw.csv', encoding='utf-8-sig')

    # 2. 숫자형 변환 (콤마 제거)
    for col in ['매출액', '영업이익']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. 반기/사업 분리
    half = df[df['보고서유형'] == '반기보고서'] \
        .set_index(['기업명', '연도'])[['매출액', '영업이익']]

    annual = df[df['보고서유형'] == '사업보고서'] \
        .set_index(['기업명', '연도'])[['매출액', '영업이익']]

    # 4. 사업보고서 = 사업보고서 - 반기보고서
    annual_adj = annual - half
    annual_adj = annual_adj.reset_index()
    annual_adj['보고서유형'] = '사업보고서'

    # 5. 반기보고서는 원본 그대로
    half_raw = half.reset_index()
    half_raw['보고서유형'] = '반기보고서'

    # 6. 합치기
    final = pd.concat([half_raw, annual_adj], ignore_index=True)

    # 7. 정렬
    final = final.sort_values(['기업명', '연도', '보고서유형'])

    # 8. 저장
    final.to_csv('./data/financial_adjust.csv', encoding='utf-8-sig', index=False)
    print('financial_adjust.csv 저장 완료')

    return final


def financial_final():
    # 1. 불러오기
    df = pd.read_csv('./data/financial_adjust.csv', encoding='utf-8-sig')

    # 숫자 변환 보정 (혹시 문자열 섞였을 경우 대비)
    for col in ['매출액', '영업이익']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 정렬 (연산 정확도를 위해)
    df = df.sort_values(['기업명', '연도', '보고서유형']).reset_index(drop=True)

    results = []

    # 기업 단위로 그룹
    for corp, g in df.groupby('기업명'):
        g = g.sort_values(['연도', '보고서유형'])

        # 보고서 구조는 일반적으로
        # (전년 사업보고서) → (해당년도 반기보고서) → (해당년도 사업보고서)
        # 순서를 기대함

        for idx, row in g.iterrows():
            year = row['연도']
            report = row['보고서유형']
            sales = row['매출액']
            op = row['영업이익']

            sales_rate = None
            op_rate = None

            if report == "반기보고서":
                # 전년도 사업보고서와 비교
                prev = g[(g['연도'] == year - 1) & (g['보고서유형'] == '사업보고서')]
                if not prev.empty:
                    prev_sales = prev.iloc[0]['매출액']
                    prev_op = prev.iloc[0]['영업이익']

                    if prev_sales != 0:
                        sales_rate = (sales - prev_sales) / prev_sales * 100
                    if prev_op != 0:
                        op_rate = (op - prev_op) / prev_op * 100

            else:  # report == "사업보고서"
                # 같은 해 반기보고서와 비교
                prev = g[(g['연도'] == year) & (g['보고서유형'] == '반기보고서')]
                if not prev.empty:
                    prev_sales = prev.iloc[0]['매출액']
                    prev_op = prev.iloc[0]['영업이익']

                    if prev_sales != 0:
                        sales_rate = (sales - prev_sales) / prev_sales * 100
                    if prev_op != 0:
                        op_rate = (op - prev_op) / prev_op * 100

            results.append({
                '기업명': corp,
                '연도': year,
                '보고서유형': report,
                '매출액': sales,
                '영업이익': op,
                '매출액증감율': sales_rate,
                '영업이익증감율': op_rate
            })

    # DataFrame 생성
    final = pd.DataFrame(results)

    # 저장
    final.to_csv('./data/financial_final.csv', encoding='utf-8-sig', index=False)
    print("financial_final.csv 저장 완료!")

    return final


def merge_emp_financial():
    import pandas as pd

    emp = pd.read_csv('./data/emp_final.csv')
    fin = pd.read_csv('./data/financial_final.csv')

    # 문자열 숫자에서 콤마 제거 후 숫자 변환
    for col in ['매출액', '영업이익']:
        if col in fin.columns:
            fin[col] = (
                fin[col]
                .astype(str)
                .str.replace(',', '', regex=False)
            )
            fin[col] = pd.to_numeric(fin[col], errors='coerce').fillna(0)

    # 병합
    merged = pd.merge(
        emp,
        fin[['기업명', '연도', '보고서유형', '매출액', '매출액증감율', '영업이익', '영업이익증감율']],
        on=['기업명', '연도', '보고서유형'],
        how='left'
    )

    merged['매출액'] = merged['매출액'].fillna(0)
    merged['영업이익'] = merged['영업이익'].fillna(0)

    merged.to_csv('./data/emp_financial_merged.csv', encoding='utf-8-sig', index=False)
    print("emp_financial_merged.csv 저장 완료!")

    return merged


def final_emp_financial():
    # 1. 파일 읽기
    df = pd.read_csv('./data/emp_financial_merged.csv', encoding='utf-8-sig')

    # 숫자형 변환 (혹시 문자열이 섞여있을 경우 대비)
    for col in ['매출액', '영업이익']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. 2016 반기보고서 제거
    df = df[~((df['연도'] == 2016) & (df['보고서유형'] == '반기보고서'))]

    # 3. 2017~2024에서 매출=0 AND 영업이익=0 인 기업 찾기
    zero_problem = (
        df[(df['연도'] >= 2017) & (df['연도'] <= 2024)]
        .groupby('기업명')
        .apply(lambda x: ((x['매출액'] == 0) & (x['영업이익'] == 0)).any())
    )

    # 문제 기업 리스트
    bad_corps = zero_problem[zero_problem].index.tolist()

    print("매출·영업이익 모두 0인 해가 존재하는 기업 수:", len(bad_corps))
    print("기업 목록:", bad_corps)

    # 4. 문제 기업 전체 데이터 제거 (2016~2024 전체 삭제)
    df_clean = df[~df['기업명'].isin(bad_corps)]

    # 5. 저장
    df_clean = df_clean.fillna(0)
    df_clean.to_csv('./data/final_emp_financial.csv', encoding='utf-8-sig', index=False)
    print("final_emp_financial.csv 저장 완료!")

    return df_clean



if __name__ == '__main__':
    # kospi_list = get_kospi_list()
    # kospi_df = match_corpcode_krx_to_dart(kospi_list)
    
    # emp_raw_df = get_emp_raw_data()
    # purify_emp_data()
    # valid_emp_data()
    transform_emp_final()
    
    # get_financial_raw_data()
    # purify_financial()
    # financial_final()
    
    merge_emp_financial()
    final_emp_financial()
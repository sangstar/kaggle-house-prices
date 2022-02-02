baseline = 2022

def get_days_ago(year):
    years_ago = baseline - year
    return years_ago * 365 # 365 is an approximation, as years don't always last 365 days
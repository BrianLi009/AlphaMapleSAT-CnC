import pandas as pd
url = r'https://kochen-specker.info/smallGraphs/'
tables = pd.read_html(url) # Returns list of all tables on page
t = tables[0] # Select table of interest
# print(t)


for index, row in t.iterrows():
    if row['Embeddability'] == "minimally unembeddable":
        print(row['Canonical name'])

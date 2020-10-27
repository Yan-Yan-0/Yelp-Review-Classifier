import requests
html = requests.get('https://www.yelp.ca/search?find_desc=Hair%20Salons&find_loc=Toronto%2C%20ON&ns=1&sortby=review_count')
from bs4 import BeautifulSoup
soup = BeautifulSoup(html.text, 'html.parser')
import pandas as pd

#Get the class that store infomation of store name and link 
results = soup.find_all("div", {'class': 'lemon--div__373c0__1mboc businessName__373c0__1fTgn display--inline-block__373c0__1ZKqC border-color--default__373c0__3-ifU'})

#Get the first 10 stores
results = results[1:11]


#Stroe all the hairsalon names 
name = []
for result in results:
    name.append(result.find('a').text)

#Get the link of each hairsalon 
link = []
for result in results:
    link.append(result.find('a').get('href'))

#Follow the link of the selected hair salon 
hs_detail = []
for hs_link in link:
    info = requests.get('https://www.yelp.ca'+hs_link)
    hs_soup = BeautifulSoup(info.text, 'html.parser')
    hs_detail.append(hs_soup)

#Create a dictonary for each hair salon with the name as key and a list of lists as value 
hairsalon = {}
for names in name:
    hairsalon[names] = []

#Function to get the rating of each review 
def get_rating(reviews):
    rate = []
    for review in reviews:
        rating = review.find('span', {'class': 'lemon--span__373c0__3997G display--inline__373c0__3JqBP border-color--default__373c0__3-ifU'})
        rating = rating.find('div').get('aria-label')
        rating = int(rating[0])
        rate.append(rating)
    return rate

#Function to get the data of each review 
def get_date(reviews):
    datepublished = []
    for review in reviews:
        date = review.find('span', {'class': 'lemon--span__373c0__3997G text__373c0__2Kxyz text-color--mid__373c0__jCeOG text-align--left__373c0__2XGa-'})
        d = date.text
        datepublished.append(d)
    return datepublished

#Function to get the content of each review 
def get_content(reviews):
    content = []
    for review in reviews:
        word = review.find('span', {'class': 'lemon--span__373c0__3997G raw__373c0__3rKqk'}).text
        content.append(word)
    return content 



for detail in hs_detail:
    reviews= detail.find_all('li', {'class': 'lemon--li__373c0__1r9wz margin-b3__373c0__q1DuY padding-b3__373c0__342DA border--bottom__373c0__3qNtD border-color--default__373c0__3-ifU'})
    reviews = reviews[:10]
    names = detail.find('h1').text

    rate = get_rating(reviews)

    datepublished = get_date(reviews)

    content = get_content(reviews)

    hairsalon[names].append(rate)
    hairsalon[names].append(datepublished)
    hairsalon[names].append(content)

#Create an empty dataframe with the required columns 
column_names = ['Name', 'RatingValue', 'DatePublished', 'Review']
df_start = pd.DataFrame(columns=column_names)

#Add the data into the dataframe 
for key in list(hairsalon.keys()):
    df = pd.DataFrame(columns=column_names)
    df['RatingValue'] = hairsalon[key][0]
    df['Name'] = key
    df['DatePublished'] = hairsalon[key][1]
    df['Review'] = hairsalon[key][2]
    df_start = df_start.append(df, ignore_index=True)

#Change the datatype
df_start = df_start.astype({'Name': 'string', 'RatingValue': 'int', 'DatePublished': 'datetime64[ns]',
'Review':'string'})
print(df_start.dtypes)



#Save the dataframe into a csv field 
df_start.to_csv('yelp_hairsalon_data_Yan.csv')
import json
import argparse
import os.path

import requests
import wikipedia
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_topics():
    soup = BeautifulSoup(
        requests.get('https://en.wikipedia.org/wiki/Wikipedia:Multiyear_ranking_of_most_viewed_pages').text)
    tr_list = soup.find('table', {'class': 'wikitable'}).find_all('tr')
    topic_list = []
    for tr in tr_list:
        td_list = tr.find_all('td')
        if len(td_list) == 0:
            continue
        if td_list[0].text.strip().isdigit():
            topic_list.append(td_list[1].text.strip())
    with open('./data/topic_list.txt', 'w') as f:
        f.writelines([i + '\n' for i in topic_list])
    return topic_list


def get_languages():
    soup = BeautifulSoup(requests.get('https://en.wikipedia.org/wiki/List_of_Wikipedias').text)
    tr_list = soup.find('table', {'class': 'wikitable plainrowheaders sortable'}).find('tbody').find_all(
        'tr')
    language_dict = {}
    for tr in tr_list:
        td_list = tr.find_all('td')
        if len(td_list) == 0:
            continue
        language_dict[td_list[2].text.strip()] = td_list[0].text.strip()
    with open('./data/languages.json', 'w') as f:
        json.dump(language_dict, f)
    return language_dict


def main(args):
    topic_list = get_topics()
    language_dict = get_languages()

    data = {}
    for i, lan in enumerate(list(language_dict.keys())[:args.topn]):
        wikipedia.set_lang(lan)
        lan_data = []
        for topic in tqdm(topic_list, desc=f'Download summaries from {language_dict[lan]} wikipedia'):
            try:
                summary = wikipedia.summary(topic)
                if summary != '':
                    lan_data.append(summary)
            except:
                continue
            # except wikipedia.exceptions.DisambiguationError as e:
            #     lan_data.append(wikipedia.summary(e.options[0]))
            #
            # except wikipedia.exceptions.PageError as e:
            #     continue
            # finally:
            #     pass
        data[lan] = lan_data
    with open(f'./data/data_{args.topn}.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topn', default=100, type=int, help='Number of top languages that will be included in the data')
    args = parser.parse_args()
    if not os.path.exists('./data'):
        os.makedirs('./data')
    main(args)
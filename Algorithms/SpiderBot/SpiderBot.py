import requests
import re


class SpiderBot:

    def __init__(self):
        self.found_urls = []    # we save url to omit repetitions
        self.iter = 1   # we track the number of found urls

    def crawl(self, first_url):

        queue = [first_url]
        self.found_urls.append(first_url)

        # breadth-first-search algorithm
        while queue:

            current_url = queue.pop(0)
            print(f"{self.iter}: {current_url}")
            self.iter += 1

            for url in self.get_urls_from_url(current_url):
                if url not in self.found_urls:
                    self.found_urls.append(url)
                    queue.append(url)

    def get_urls_from_url(self, url):
        try:
            html = requests.get(url, timeout=10).text
        except:
            return []

        return re.findall(r"https?://[\w.-]+\.[a-z]{2,3}", html)
        """ 
            The regex finds urls in the html. However, its form is pretty comprehensive and
            not ideal but it works good enough.
        """

if __name__ == '__main__':

    crawler = SpiderBot()
    crawler.crawl('https://stackoverflow.com')


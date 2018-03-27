
import requests
# from pyquery import PyQuery as pq
from lxml.html import fromstring

uri = "https://utslogin.nlm.nih.gov"
# option 1 - username/pw authentication at /cas/v1/tickets
# auth_endpoint = "/cas/v1/tickets/"
# option 2 - api key authentication at /cas/v1/api-key
auth_endpoint = "/cas/v1/api-key"

'''

'''
class Authentication:
    '''
    The UMLS REST API requires a UMLS account for the authentication

    You can find the API key in the UTS ‘My Profile’ area after signing in. An API key remains active as long as the associated UTS account is active.
    If the API key field in your UTS profile is blank, click ‘Edit Profile’, select the ‘Generate new API Key’ checkbox, then scroll down and click ‘Save Profile’.
    https://uts.nlm.nih.gov/home.html#profile

    '''

    # def __init__(self, username,password):
    def __init__(self, apikey):
        self.apikey = apikey
        self.service = "http://umlsks.nlm.nih.gov"

    def gettgt(self):
        params = {'apikey': self.apikey}
        h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(uri + auth_endpoint, data=params, headers=h)
        response = fromstring(r.text)
        ## extract the entire URL needed from the HTML form (action attribute) returned - looks similar to https://utslogin.nlm.nih.gov/cas/v1/tickets/TGT-36471-aYqNLN2rFIJPXKzxwdTNC5ZT7z3B3cTAKfSc5ndHQcUxeaDOLN-cas
        ## we make a POST call to this URL in the getst method
        tgt = response.xpath('//form/@action')[0]
        return tgt

    def getst(self, tgt):
        params = {'service': self.service}
        h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(tgt, data=params, headers=h)
        st = r.text
        return st

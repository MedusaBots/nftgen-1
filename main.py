from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import FirefoxOptions
from webdriver_manager.chrome import ChromeDriverManager
opts = FirefoxOptions()
opts.add_argument("--headless")
driver = webdriver.Firefox(service=Service("/usr/bin/geckodriver"), options=opts)

ids = driver.find_element_by_class_name('option.seedTextBox')
print(ids)
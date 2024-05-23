#!/usr/bin/env python3
import cgi

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(credentials)

# Google Spreadsheet details
spreadsheet_key = 'your_spreadsheet_key'
sheet_name = 'Sheet1'  # Replace with your sheet name

# Function to append row to Google Spreadsheet
def append_row_to_sheet(name, email, feedback):
    try:
        sheet = client.open_by_key(spreadsheet_key).worksheet(sheet_name)
        row = [name, email, feedback]
        sheet.append_row(row)
        return True
    except Exception as e:
        print(str(e))
        return False

# Print HTTP headers
print("Content-type: text/html\n")

# Print HTML response
print("<html><head><title>Feedback Submitted</title></head><body>")
print("<h2>Feedback Submitted Successfully!</h2>")

# Get form data
form = cgi.FieldStorage()

name = form.getvalue('name')
email = form.getvalue('email')
feedback = form.getvalue('feedback')

# Print feedback details
print("<p><strong>Name:</strong> {}</p>".format(name))
print("<p><strong>Email:</strong> {}</p>".format(email))
print("<p><strong>Feedback:</strong><br> {}</p>".format(feedback))

# Append row to Google Spreadsheet
if append_row_to_sheet(name, email, feedback):
    print("<p>Thank you for your feedback!</p>")
else:
    print("<p>Sorry, there was a problem storing your feedback.</p>")

print("<p><a href='javascript:window.close()'>Close this window</a></p>")
print("</body></html>")

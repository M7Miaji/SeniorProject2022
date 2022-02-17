import sqlite3
#connection with database
con = sqlite3.connect('db.sqlite3')
#cursor mean like pointer
cursor = con.cursor()
#print all names in master db
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
cursor.execute("SELECT * FROM client;")
print(cursor.fetchall())
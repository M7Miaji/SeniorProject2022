from multiprocessing.sharedctypes import Value
from readline import insert_text
import sqlite3
#connection with database
con = sqlite3.connect('db.sqlite3')
#cursor mean like pointer
cursor = con.cursor()
#print all names in master db
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(cursor.fetchall())
#cursor.execute("SELECT * FROM client;")
#print(cursor.fetchall())
#INSERT INTO user (f-name,l-name,email,password,admin)
#Value ("ysf,tgf,yf@h.v,12345,1");
user= ('Khaled,Alkerithi,khaledmosel@gmail.com,Yousefisalegend,0'); 
def create_user(con, user):
    """
    Create a new user
    :param con:
    :param user:
    :return: user-id
    """
    sql = ''' INSERT INTO user(f-name,l-name,email,password,admin)
              VALUES(?,?,?,?,?) '''
    cur = con.cursor()
    cur.execute(sql, user)
    con.commit()
    return cur.lastrowid
create_user(con,user)
print(cursor.fetchall())

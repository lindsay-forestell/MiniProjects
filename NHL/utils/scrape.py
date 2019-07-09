# import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import unidecode


class HTMLTableParser:
    """
    Taken from https://srome.github.io/Parsing-HTML-Tables-in-Python-with-BeautifulSoup-and-pandas/
    Returns a list of tables from a given URL
    """

    def parse_url(self, url):
        """
        Parses the URL into beautiful soup and then returns all of the tables.
        :param url: url of interest
        :return: (id, table) list
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        try:
            idx = [table['id'] for table in soup.find_all('table')]
        except KeyError:
            idx = [0]*len(soup.find_all('table'))
        return [(idx[i], self.parse_html_table(table))
                for i, table in enumerate(soup.find_all('table'))]

    def parse_html_table(self, table):
        """
        Internal function that actually separates the tables
        :param table: for a table passed in, returns as a dataframe
        :return: table df
        """
        n_columns = 0
        n_rows = 0
        column_names = []

        # Find number of rows and columns
        # we also find the column titles if we can
        for row in table.find_all('tr'):

            # Determine the number of rows in the table
            td_tags = row.find_all('td')
            if len(td_tags) > 0:
                n_rows += 1
                if n_columns == 0:
                    # Set the number of columns for our table
                    n_columns = len(td_tags)

            # Handle column names if we find them
            th_tags = row.find_all('th')
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text())

        # Safeguard on Column Titles
        if len(column_names) > 0 and len(column_names) != n_columns:
            raise Exception("Column titles do not match the number of columns")

        columns = column_names if len(column_names) > 0 else range(0, n_columns)
        df = pd.DataFrame(columns=columns,
                          index=range(0, n_rows))
        row_marker = 0
        for row in table.find_all('tr'):
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                df.iat[row_marker, column_marker] = column.get_text()
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1

        # Convert to float if possible
        for col in df:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

        return df


class PlayerContractParser:
    """
    class that parses player contracts specifically
    """
    def __init__(self, player='Connor McDavid'):

        self.months = {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9,
            'October': 10,
            'November': 11,
            'December': 12
        }
        self.player = player.lower().replace(' ', '-')
        # check for first initial players like p k  subban
        name_list = self.player.split('-')
        if (len(name_list[0]) == 1) & (len(name_list[1]) == 1):
            self.player = name_list[0] + name_list[1] + '-'
            for i in range(2, len(name_list)):
                if name_list[i] != '':
                    self.player += name_list[i] + '-'
            self.player = self.player[:-1]

        # website where info will come from
        self.url = 'https://www.capfriendly.com/players/'

        self.contracts = {
            'contract_type': [],
            'length': [],
            'cap_hit_pct': [],
            'signing_status': [],
            'signing_age': [],
            'signing_date': []
        }
        self.contracts_df = None

        self.response = requests.get(self.url + self.player)
        self.soup = BeautifulSoup(self.response.text, 'lxml')

        # some of the player pages don't exist, but do with player-name1 instead of player-name. So try this url too.
        try:
            date = self.soup.find(attrs={'class': 'indx_b l'}).find('div').get_text().strip('BORN:').strip(' ')
            self.age = self.get_date(date)
            self.get_contract_tables()
            self.contracts_df = pd.DataFrame(self.contracts)
        except AttributeError:
            # try adding a 1 to the end of the name to see if that html works
            self.player = self.player + '1'
            self.response = requests.get(self.url + self.player)
            self.soup = BeautifulSoup(self.response.text, 'lxml')
            try:
                date = self.soup.find(attrs={'class': 'indx_b l'}).find('div').get_text().strip('BORN:').strip(' ')
                self.age = self.get_date(date)
                self.get_contract_tables()
                self.contracts_df = pd.DataFrame(self.contracts)
            except AttributeError:
                print('Player {:s} Not Found.'.format(player))

    # Return a tuple corresponding to (day, month, year)
    def get_date(self, date):
        """
        Given a date, return it as a numerical tuple
        :param date: In the form "February 29, 2042"
        :return: numerical tuple of the date
        """
        date_list = date.split()
        month = self.months[date_list[0]]
        day = int(date_list[1].strip(','))
        year = int(date_list[2])
        return day, month, year

    def get_contract_tables(self):
        """
        get all the appropriate tables for a given skater
        :return: nothing, but will update the contract lists inside the class.
        """
        tables = self.soup.find_all(attrs={'class': 'table_c'})

        # If they currently have a contract, it shows up at the top.
        # Append it to the bottom so contracts are in order from earliest -> latest.
        if 'CURRENT CONTRACT' in [item.get_text() for item in self.soup.find_all('h4')]:
            tables.append(tables[0])
            tables.pop(0)

        # All players start as UFA's
        signing_status = 0

        # ...Unless they are old. Then they have a 'historical' section that needs to be skipped
        # And their status set to UFA.
        for table in tables:

            try:
                check_historical = 'HISTORICAL SALARY' in table.find_all(attrs={'class': 'ofh'})[0].get_text()
                if check_historical:
                    signing_status = 1
                    continue
            except IndexError:
                continue

            contract_type = table.find('h6').get_text()

            # First line contains contract length, status at expiration
            first_line = table.find_all(attrs={'class': 'l cont_t mt4 mb2'})
            length = int(first_line[0].get_text().split(':')[1].split()[0])
            new_status = first_line[1].get_text().split(':')[1].split()[0]
            if new_status == 'RFA':
                new_status = 0
            else:
                new_status = 1

            # Second line contains cap hit and signing date
            second_line = table.find_all(attrs={'class': 'l cont_t mb5'})
            cap_hit_pct = float(second_line[1].get_text().split(':')[1].split()[0]) / 100
            sign_date = second_line[2].get_text().split(':')[1].strip(' ')
            try:
                sign_date_tuple = self.get_date(sign_date)
            except KeyError:
                print('Player {:s}, Sign Date {:s}'.format(self.player, sign_date))
                continue

            # Figure out their age at signing
            # Reduce by a year if their birthday hasn't happened yet.
            signing_age = sign_date_tuple[2] - self.age[2]
            if (self.age[0] + self.age[1] * 1000) > (sign_date_tuple[0] + sign_date_tuple[1] * 1000):
                signing_age -= 1

            # Append the player contract to their list of past/present contracts.
            self.contracts['contract_type'].append(contract_type)
            self.contracts['length'].append(length)
            self.contracts['signing_status'].append(signing_status)
            self.contracts['cap_hit_pct'].append(cap_hit_pct)
            self.contracts['signing_age'].append(signing_age)
            self.contracts['signing_date'].append(sign_date_tuple)

            # Update their new status to be that that it will be when their contract expires
            signing_status = new_status


def rename_player(player):
    """
    Rename a player from "First Last" to "FIRST.LAST". Also removes accents if necessary.
    :param player: player name
    :return: player name, formatted
    """
    try:
        name_split = player.split(' ')
        if len(name_split) == 1:
            return player
        name_split = name_split[1:]
        player = ''
        for name in name_split:
            player += unidecode.unidecode(name).upper() + '.'
        return player[:-1]
    except Exception as e:
        print("Error at player:", player)
        print(e)


def get_date(date):
    """
    Determine the date as a numerical tuple
    :param date: Date in February 29, 2042 format.
    :return: Date in (day, month, year) format.
    """
    months = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12,
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }
    date_list = date.split()
    month = months[date_list[0]]
    day = int(date_list[1].strip(','))
    year = int(date_list[2])
    return day, month, year

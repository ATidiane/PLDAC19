#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import MySQLdb
import MySQLdb.converters
from MySQLdb.constants import FIELD_TYPE
from MySQLdb.cursors import SSCursor


class Connection:
    """
    Connects to Drunk ``stif`` database via MySQLdb.
    """

    def __init__(self):
        conversions = MySQLdb.converters.conversions.copy()
        conversions[FIELD_TYPE.DATE] = str  # STIF dates in str because Y-M-D
        conversions[FIELD_TYPE.TIME] = str  # STIF times in str because H-M-S

        self._connection = MySQLdb.connect(
            host='drunk',
            user='stif',
            passwd='Zjk7lJ0oqfuxy85G',
            db='stif',
            conv=conversions,
            local_infile=True
        )

        print('Drunk connection opened')

    def __del__(self):
        self._connection.commit()
        self._connection.close()

        print ('Drunk connection closed')

    def cursor(self):
        return self._connection.cursor()

    def sscursor(self):
        return SSCursor(self._connection)

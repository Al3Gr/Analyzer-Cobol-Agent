       IDENTIFICATION DIVISION.
        PROGRAM-ID. PRODUCTS.
        ENVIRONMENT DIVISION.
        INPUT-OUTPUT SECTION.
        FILE-CONTROL.
           SELECT IN-FILE
           ASSIGN TO ".\products.txt"
                ORGANIZATION IS LINE SEQUENTIAL.

        DATA DIVISION.
        FILE SECTION.
        FD IN-FILE.
       01 IN-RECORD.
          05 PROD-CODE    PIC X(9).
          05 PROD-NAME    PIC X(37).
          05 PROD-UNIT    PIC X(7).
          05 PROD-PRICE   PIC S9(4)V9(2).

        WORKING-STORAGE SECTION.
       01 SWITCHES.
          05 EOF-SWITCH   PIC X          VALUE "N".

       01 COUNTERS.
          05 REC-COUNTER  PIC 9(3)       VALUE 0.

       01 ORD             PIC 9(2).
       01 ORD-T.
          05 PCODE        PIC X(9) OCCURS 100 TIMES.
          05 PNAME        PIC X(37) OCCURS 100 TIMES.
          05 PUNIT        PIC X(7) OCCURS 100 TIMES.
          05 PPRICE       PIC 9(4)V9(2) OCCURS 100 TIMES.
       01 TOTAL           PIC 9(4)V9(2).
       01 MONEY           PIC 9(4)V9(2).
       01 CHANGE          PIC 9(4)V9(2).
       01 I               PIC 9(2)       VALUE 0.
        PROCEDURE DIVISION.
       000-MAIN.
           PERFORM 100-INITIALIZE.
           DISPLAY "PRODUCTS".
           PERFORM 200-PROCESS-RECORDS
              UNTIL EOF-SWITCH = "Y".
           PERFORM 300-TERMINATE.
           PERFORM ORDER-FORM.
           STOP RUN.

       100-INITIALIZE.
           OPEN INPUT IN-FILE.

           READ IN-FILE
           AT END
              MOVE "Y" TO EOF-SWITCH
           NOT AT END
               COMPUTE REC-COUNTER = REC-COUNTER + 1
           END-READ.

       200-PROCESS-RECORDS.

           DISPLAY IN-RECORD.


           READ IN-FILE
           AT END
              MOVE "Y" TO EOF-SWITCH
           NOT AT END
               COMPUTE REC-COUNTER = REC-COUNTER + 1
           END-READ.

       300-TERMINATE.
           DISPLAY "NUM OF PRODUCTS >>>>> " REC-COUNTER.
           CLOSE IN-FILE.

       ORDER-FORM.
           DISPLAY "HOW MANY PRODUCTS TO ORDER ?"
           ACCEPT ORD.
           MOVE 1 TO I.
           PERFORM UNTIL I > ORD
                   DISPLAY "ENTER PRODUCT CODE : "
                   ACCEPT PCODE(I)
                   DISPLAY "ENTER PRODUCT NAME : "
                   ACCEPT PNAME(I)
                   DISPLAY "ENTER PRODUCT UNIT : "
                   ACCEPT PUNIT(I)
                   DISPLAY "ENTER PRODUCT PRICE : "
                   ACCEPT PPRICE(I)
                   ADD 1 TO I
           END-PERFORM.

           DISPLAY "--------------------------------------------------"
           DISPLAY "|                         RECEIPT                |"
           DISPLAY "-------------------------------------------------"
           MOVE 1 TO I
           PERFORM UNTIL I > ORD
                   DISPLAY PCODE(I), PNAME(I), PUNIT(I), PPRICE(I)
                   ADD 1 TO I
           END-PERFORM.
           DISPLAY "-------------------------------------------------"
           MOVE 1 TO I
           PERFORM UNTIL I > ORD
                   COMPUTE TOTAL = TOTAL + PPRICE(I)
                   ADD 1 TO I
           END-PERFORM.
           DISPLAY "                                           " TOTAL.
           DISPLAY "-------------------------------------------------".
           DISPLAY " ENTER PAYMENT : ".
           ACCEPT MONEY.
           COMPUTE CHANGE = MONEY - TOTAL.
           DISPLAY "-------------------------------------------------".
           DISPLAY "                                           " MONEY.
           DISPLAY "                                           " TOTAL.
           DISPLAY "CHANGE :                                  " CHANGE.
           DISPLAY "-------------------------------------------------".
           DISPLAY "|           THANK YOU FOR SHOPPING!             |".
           DISPLAY "--------------------------------------------------".
           STOP RUN.
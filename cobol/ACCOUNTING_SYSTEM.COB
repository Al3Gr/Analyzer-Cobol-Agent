
      *
      ******************************************************************
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SUPERMARKET-SYSTEM.
       AUTHOR. BSIT 2-3.
       INSTALLATION. DESKTOP.
       DATE-WRITTEN. FEBRUARY 11, 2023.
       DATE-COMPILED. FEBRUARY 18, 2023.
       SECURITY. WALA LODS.
       REMARKS. PROJECT.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. DESKTOP.
       OBJECT-COMPUTER. DESKTOP.

       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
            SELECT INFILE ASSIGN TO ".\DATABASE.txt".
            SELECT OUTFILE ASSIGN TO ".\DATABASE_1.TXT".



       DATA DIVISION.

       FILE SECTION.
       FD   INFILE
             LABEL RECORDS ARE STANDARD
             RECORD CONTAINS 60 CHARACTERS
             DATA RECORD IS REC-IN.
       01   REC-IN.
            05 PCODE-IN PIC 9(8).
            05 PNAME-IN PIC X(37).
            05 UNIT-IN PIC X(7).
            05 PRICE-IN PIC 9(8).

       FD   OUTFILE.


       01   OUTREC PIC X(60).



       WORKING-STORAGE SECTION.

       01   REC-OUT.
            05 PCODE-OUT PIC 9(8).
            05 PNAME-OUT PIC X(37).
            05 PUNIT-OUT PIC X(7).
            05 PRICE-OUT PIC 9(6)V99.

       01   TEMP-VARIBLES.
            05 TOKEN PIC 9.
            05 TEMP-PRODUCT-CODE PIC 9(6).

       01   MAIN-VAR.
            05 PCODE PIC 9(8).

            05 PNAME PIC X(37).
            05 PUNIT PIC X(7).
            05 PRICE PIC 9(6)V99.


            05 PRODUCT-CODE PIC 9(8).
            05 PRODUCT-NAME PIC X(20).

            05 TOT-AMOUNT PIC 9(6)V99 VALUE 0.

            05 TOT-TOTAL PIC 9(6)V99 VALUE 0.

       01   ADMIN-INFO.
            05 EMAIL PIC X(15) VALUE "robby@gmail.com".
            05 ADMIN-PASSWORD PIC X(9) VALUE "robby@123".

       01   LOG-INFO.
            05 EMAIL-IN PIC X(60).
            05 PASSWORD-IN PIC X(10).

       01   CHOICE PIC 9.
       01   ANSWER PIC X.

       01   EOFSW PIC X VALUE 'N'.
       01   REC-COUNTER PIC 9(3) VALUE 0.
       01   DELETE-PRODUCT-CODE PIC 9(8).

       01  EMPLOYE-INFO.
            05 EMPLOYEE-NAME PIC X(50).
            05 EMPLOYEE-ID PIC X(10).
            05 HOURLY-RATE PIC 9(5) VALUE 500.
            05 HOURS-WORKED PIC 99.
            05 SALARY PIC 9(6).
       01  PROFIT-INFO.
            05 COGS PIC 9(6)V99.
            05 SELLING-PRICE PIC 9(6)V99.
            05 PROFIT PIC 9(6)V99.
       01  ORD PIC 9(2).
       01  ORD-T.
             05 PPCODE PIC X(9) OCCURS 100 TIMES.
             05 PPNAME PIC X(37) OCCURS 100 TIMES.
             05 PPUNIT PIC X(7) OCCURS 100 TIMES.
             05 PPRICE PIC 9(4)V99 OCCURS 100 TIMES.
       01   TOTAL PIC 9(4)V9(2).
       01   MONEY PIC 9(4)V9(2).
       01   CHANGE PIC 9(4)V9(2).
       01   I PIC 9(2) VALUE 0.



       SCREEN SECTION.
       01   CLEARSCREEN.
            05 BLANK SCREEN.


       PROCEDURE DIVISION.

       MAIN-ROUTINE.
           OPEN INPUT INFILE OUTPUT OUTFILE.

             PERFORM UNTIL  EOFSW = 'Y'
             READ INFILE
               AT END
                 MOVE 'Y' TO EOFSW
               NOT AT END
                 MOVE REC-IN  TO REC-OUT
                 WRITE OUTREC FROM REC-OUT
                 IF DELETE-PRODUCT-CODE NOT EQUAL TO PCODE-IN
                     WRITE OUTREC FROM REC-OUT
                 END-IF
             END-READ
            END-PERFORM.


           PERFORM INIT-ROUTINE THROUGH INIT-END.

           CLOSE INFILE OUTFILE .

       INIT-ROUTINE.

           DISPLAY CLEARSCREEN.


           DISPLAY  (5, 25)"SUPERMARKET MAIN-MENU".
           DISPLAY  (8, 26)" 1) ADMINISTRATOR".
           DISPLAY  (9, 26)" 2) BUYER".
           DISPLAY (10, 26)" 3) EXIT".
           DISPLAY (12, 26)"  PLEASE  SELECT".
           ACCEPT  (12, 47) CHOICE.

           IF CHOICE = 1 THEN
             PERFORM LOG-ROUTINE THROUGH LOG-END

           ELSE IF CHOICE = 2 THEN
             PERFORM BUYER-ROUTINE THROUGH BUYER-END
           ELSE IF CHOICE = 3 THEN
             STOP RUN
           ELSE
             PERFORM INIT-ROUTINE THROUGH INIT-END
           END-IF.

       INIT-END.


       SHOPPING-ROUTINE.
       SHOPPING-END.

       LOG-ROUTINE.
            DISPLAY CLEARSCREEN.

            DISPLAY  (6, 25) "PLEASE LOG IN ".
            DISPLAY  (7, 25) "ENTER EMAIL :".
            ACCEPT   (7, 39) EMAIL-IN.
            DISPLAY  (8, 25) "ENTER PASSWORD :".
            ACCEPT   (8, 43) PASSWORD-IN WITH SECURE.



            IF EMAIL-IN = EMAIL AND PASSWORD-IN = ADMIN-PASSWORD THEN
              PERFORM ADMISNISTRATOR-ROUTINE THROUGH ADMINISTRATOR-END
            ELSE
              DISPLAY (11, 25) "INVALID EMAIL AND PASSWORD!"


              PERFORM  INVALID-ROUTINE THROUGH INVALID-END.
       LOG-END.

       INVALID-ROUTINE.
             DISPLAY (13, 25) "0 TRY AGAIN".
             DISPLAY (14, 25) "1 BACK".
             ACCEPT  (17, 25) CHOICE.

             IF CHOICE = 1 THEN
                   PERFORM INIT-ROUTINE THROUGH INIT-END
             ELSE IF  CHOICE = 0 THEN
                   PERFORM LOG-ROUTINE THROUGH LOG-END
             ELSE
                   DISPLAY (16, 25) "INVALID INPUT"
                   PERFORM INVALID-ROUTINE THROUGH INVALID-END
            END-IF.
       INVALID-END.

       ADMISNISTRATOR-ROUTINE.
            DISPLAY CLEARSCREEN.

            DISPLAY  (5, 25)" ADMINISTRATOR  MENU".
            DISPLAY  (8, 25)"1) EMPLOYEE SALARY".
            DISPLAY  (9, 25)"2) MODIFY THE PRODUCT".
            DISPLAY (10, 25)"3) SUPERMARKET PROFIT".
            DISPLAY (11, 25)"4) BACK TO MAIN MENU".
            DISPLAY (13, 25)"   PLEASE  SELECT".
            ACCEPT  (13, 47) CHOICE.

            IF CHOICE = 1 THEN
              PERFORM SALARY-ROUTINE THROUGH SALARY-END
            ELSE IF  CHOICE = 2 THEN
              PERFORM MODIFY-ROUTINE THROUGH MODIFY-END
            ELSE IF CHOICE = 3 THEN
              PERFORM PROFIT-ROUTINE THROUGH PROFIT-END
            ELSE IF CHOICE = 4 THEN
              PERFORM INIT-ROUTINE THROUGH INIT-END
            ELSE
              PERFORM ADMISNISTRATOR-ROUTINE THROUGH ADMINISTRATOR-END
            END-IF.

       ADMINISTRATOR-END.

       BUYER-ROUTINE.
            DISPLAY CLEARSCREEN.

            DISPLAY  (5, 25)"        BUYER          ".
            DISPLAY  (8, 25)" 1) BUY PRODUCT        ".
            DISPLAY  (9, 25)" 2) GO BACK            ".
            DISPLAY (10, 25)"     PLEASE  SELECT    ".
            ACCEPT  (10, 48) CHOICE.

            IF CHOICE = 1 THEN
              PERFORM LIST-ROUTINE THROUGH LIST-END
            ELSE IF CHOICE = 2 THEN
              PERFORM INIT-ROUTINE THROUGH INIT-END
            ELSE
              PERFORM BUYER-ROUTINE THROUGH BUYER-END
            END-IF.


       BUYER-END.

       BUY-ROUTINE.
             DISPLAY CLEARSCREEN.

             PERFORM LIST-ROUTINE  THROUGH LIST-END.

       BUY-END.

       ADD-ROUTINE.
            DISPLAY CLEARSCREEN.

            DISPLAY  (5, 25)"ADD NEW PRODUCT".

            DISPLAY  (8, 25)" PRODUCT CODE :".
            ACCEPT   (8, 42) PCODE.

            DISPLAY  (9, 25)" PRODUCT NAME :".
            ACCEPT   (9, 42) PNAME.

            DISPLAY (10, 25)" UNIT         :".
            ACCEPT  (10, 42)PUNIT.

            DISPLAY (11, 25)" PRICE        :".
            ACCEPT  (11, 42)PRICE.

            MOVE PCODE TO PCODE-OUT.
            MOVE PNAME TO PNAME-OUT.
            MOVE PUNIT TO PUNIT-OUT.
            MOVE PRICE TO PRICE-OUT.



            WRITE OUTREC FROM REC-OUT.



            DISPLAY (12, 25)" PRODUCT RECORDED!".

            PERFORM CONTINUE-ROUTINE THROUGH CONTINUE-END.



       ADD-END.


        CONTINUE-ROUTINE.
            DISPLAY (13, 25)" ENTER ANOTHER? [Y/N]".
            ACCEPT  (13, 48) ANSWER.

           IF ANSWER = 'Y' OR ANSWER = 'y' THEN
              PERFORM ADD-ROUTINE THROUGH ADD-END
            ELSE IF ANSWER = 'N' OR ANSWER = 'n' THEN
              PERFORM MODIFY-ROUTINE THROUGH MODIFY-END
            ELSE
              DISPLAY (14, 25)"INVALID INPUT"
              PERFORM CONTINUE-ROUTINE THROUGH CONTINUE-END
            END-IF.


       CONTINUE-END.

       MODIFY-ROUTINE.
            DISPLAY CLEARSCREEN.

            DISPLAY (5, 25) "MODIFY MENU".

            DISPLAY (7, 25) "1 ADD PRODUCT".
            DISPLAY (8, 25) "2 DELETE PRODUCT".
            DISPLAY (9, 25) "3 BACK".
            DISPLAY (10, 27) "PLEASE SELECT:".
            ACCEPT (10, 43) CHOICE.

            IF CHOICE = 1 THEN
              PERFORM ADD-ROUTINE THROUGH ADD-END
            ELSE IF CHOICE = 2 THEN
              PERFORM REMOVE-ROUTINE THROUGH REMOVE-END
            ELSE IF CHOICE = 3 THEN
              PERFORM ADMISNISTRATOR-ROUTINE THROUGH ADMINISTRATOR-END
            ELSE
              PERFORM MODIFY-ROUTINE THROUGH MODIFY-END
            END-IF.



       MODIFY-END.


       REMOVE-ROUTINE.

           DISPLAY CLEARSCREEN.
           DISPLAY "ENTER PRODUCT CODE TO DELETE: ".
           ACCEPT DELETE-PRODUCT-CODE.

           PERFORM UNTIL EOFSW = 'Y'
           READ INFILE
             AT END
                 MOVE 'Y' TO EOFSW
             NOT AT END
                 IF DELETE-PRODUCT-CODE NOT EQUAL TO PCODE-IN
                     MOVE REC-IN TO REC-OUT
                     WRITE OUTREC FROM REC-OUT
                 END-IF
           END-READ
           END-PERFORM.
       REMOVE-END.

       SEARCH-PRODUCT-ROUTINE.
            MOVE 'N' TO ANSWER.

            PERFORM UNTIL EOFSW = 'Y'
               READ INFILE
                   AT END MOVE 'Y' TO EOFSW
               END-READ

            IF EOFSW = 'N' AND TEMP-PRODUCT-CODE = PCODE-IN THEN
              DISPLAY (10, 25) "PRODUCT FOUND!"
              DISPLAY (11, 25) "PRODUCT CODE : " PCODE-IN
              DISPLAY (12, 25) "PRODUCT NAME : " PNAME-IN
              DISPLAY (13, 25) "PRODUCT UNIT : " UNIT-IN
              DISPLAY (14, 25) "PRODUCT PRICE: " PRICE-IN

              DISPLAY (15, 25) "DELETE THIS PRODUCT? [Y/N]"

              ACCEPT (15, 53) ANSWER

              MOVE 'Y' TO EOFSW
             END-IF

             END-PERFORM.
       SEARCH-PRODUCT-END.






       LIST-ROUTINE.
            DISPLAY CLEARSCREEN.

            PERFORM INITIALIZE-ROUTINE THROUGH INITIALIZE-END.
            DISPLAY "PRODUCTS".
            PERFORM PROCESS-RECORD-ROUTINE THROUGH PROCESS-RECORD-END
            UNTIL EOFSW = 'Y'.
            PERFORM TERMINATE-ROUTINE THROUGH TERMINATE-END.
            PERFORM RECEIPT-ROUTINE THROUGH RECEIPT-END.

            DISPLAY 'tapos na'.



       LIST-END.

       INITIALIZE-ROUTINE.
            DISPLAY CLEARSCREEN.


            READ INFILE
            AT END
               MOVE 'Y' TO EOFSW
            NOT AT END
               COMPUTE REC-COUNTER = REC-COUNTER + 1
            END-READ.
       INITIALIZE-END.

       PROCESS-RECORD-ROUTINE.
             DISPLAY REC-IN.

             READ INFILE
             AT END
             MOVE 'Y' TO EOFSW
             NOT AT END

            COMPUTE REC-COUNTER = REC-COUNTER + 1
               END-READ.
       PROCESS-RECORD-END.

       TERMINATE-ROUTINE.
            DISPLAY "NUM OF PRODUCTS >>>>> " REC-COUNTER.

       TERMINATE-END.

       RECEIPT-ROUTINE.
            DISPLAY CLEARSCREEN.

             DISPLAY "HOW MANY PRODUCTS TO ORDER ?"
           ACCEPT ORD.
           MOVE 1 TO I.
           PERFORM UNTIL I > ORD
               DISPLAY "ENTER PRODUCT CODE : "
               ACCEPT PPCODE(I)
               DISPLAY "ENTER PRODUCT NAME : "
               ACCEPT PPNAME(I)
               DISPLAY "ENTER PRODUCT UNIT : "
               ACCEPT PPUNIT(I)
               DISPLAY "ENTER PRODUCT PRICE : "
               ACCEPT PPRICE(I)
               ADD 1 TO I
           END-PERFORM.

            DISPLAY "-------------------------------------------------"
            DISPLAY "|                      RECEIPT                  |"
            DISPLAY "-------------------------------------------------"
            MOVE 1 TO I
             PERFORM UNTIL I > ORD
            DISPLAY PPCODE(I), PPNAME(I), PPUNIT(I), PPRICE(I)
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
            DISPLAY "-------------------------------------------------".
       RECEIPT-END.


       SALARY-ROUTINE.
           DISPLAY CLEARSCREEN.

           DISPLAY (5, 35) "SALARY ".

           DISPLAY (7, 25) "ENTER EMPLOYEE NAME: ".
           ACCEPT EMPLOYEE-NAME.

           DISPLAY (8, 25)"ENTER EMPLOYEE ID: ".
           ACCEPT EMPLOYEE-ID.

           DISPLAY (10, 25)"ENTER HOURS WORKED: ".
           ACCEPT HOURS-WORKED.

           COMPUTE SALARY = HOURLY-RATE * HOURS-WORKED.

           DISPLAY (12, 25)"EMPLOYEE NAME: ", EMPLOYEE-NAME.
           DISPLAY (13, 25)"EMPLOYEE ID: ", EMPLOYEE-ID.
           DISPLAY (14, 25)"HOURLY RATE: PHP ", HOURLY-RATE.
           DISPLAY (15, 25)"HOURS WORKED: ", HOURS-WORKED.
           DISPLAY (16, 25)"SALARY: PHP ", SALARY.



           PERFORM SALARY-CONTINUE-ROUTINE THROUGH SALARY-CONTINUE-END.

       SALARY-END.

       SALARY-CONTINUE-ROUTINE.
           DISPLAY (17, 25) "ENTER ANOTHER? [Y/N]".
           ACCEPT (17, 47) ANSWER.

           IF ANSWER = 'Y' OR ANSWER = 'y' THEN
             PERFORM SALARY-ROUTINE THROUGH SALARY-END
           ELSE IF ANSWER = 'N' OR  ANSWER = 'n' THEN
             PERFORM ADMISNISTRATOR-ROUTINE THROUGH ADMINISTRATOR-END
           ELSE
             DISPLAY (18, 25) "INVALID INPUT"
             PERFORM SALARY-CONTINUE-ROUTINE
             THROUGH SALARY-CONTINUE-END
           END-IF.

       SALARY-CONTINUE-END.

       PROFIT-ROUTINE.
           DISPLAY CLEARSCREEN.

           DISPLAY (5, 25) "PROFIT".
           DISPLAY (8, 25)"CALCULATION OF PROFIT".
           DISPLAY (9, 25)"ENTER COST  OF GOODS SOLD: ".
           ACCEPT (9, 52)COGS.
           DISPLAY (10, 25)"ENTER  SELLING PRICE: ".
           ACCEPT (10, 52) SELLING-PRICE.
           COMPUTE PROFIT = SELLING-PRICE - COGS.
           DISPLAY (11, 25)"PROFIT: ", PROFIT.


           PERFORM PROFIT-CONTINUE-ROUTINE THROUGH PROFIT-CONTINUE-END.

       PROFIT-END.

       PROFIT-CONTINUE-ROUTINE.
           DISPLAY (12, 25) "ENTER ANOTHER? [Y/N]".
           ACCEPT  (12, 47) ANSWER.

           IF ANSWER = 'Y' OR ANSWER = 'y' THEN
             PERFORM PROFIT-ROUTINE THROUGH PROFIT-END
           ELSE IF ANSWER = 'N' OR  ANSWER = 'n' THEN
             PERFORM ADMISNISTRATOR-ROUTINE THROUGH ADMINISTRATOR-END
           ELSE
             DISPLAY (18, 25) "INVALID INPUT"
             PERFORM PROFIT-CONTINUE-ROUTINE
             THROUGH PROFIT-CONTINUE-END
           END-IF.
       PROFIT-CONTINUE-END.
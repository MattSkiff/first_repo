REM "C:\Program Files\R\R-3.5.2\bin\R.exe" CMD BATCH H:\finite_difference_flow_fields.R
@echo off
C:
PATH C:\Programme\R\R-3.0.1\bin;%path%
cd H:\
Rscript finite_difference_flow_fields.R
pause

from pydantic import BaseModel

class LoanPred(BaseModel):
    current_loan_amount: int
    term: str 
    credit_score: float 
    years_in_current_job: float  
    home_ownership: str  
    annual_income: float 
    purpose: str 
    monthly_debt: float
    years_of_credit_history: float 
    months_since_last_delinquent: float
    number_of_open_accounts: int
    number_of_credit_problems: int
    current_credit_balance: int
    maximum_open_credit: float
    bankruptcies: float
    tax_liens: float
        
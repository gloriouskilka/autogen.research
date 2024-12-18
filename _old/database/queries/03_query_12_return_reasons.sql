-- Track Return Rates and Reasons

SELECT
    b.BookID,
    b.Title,
    COUNT(r.ReturnID) AS TotalReturns,
    SUM(r.QuantityReturned) AS TotalUnitsReturned,
    ROUND((SUM(r.QuantityReturned) / NULLIF(SUM(-a.QuantityAdjusted), 0)) * 100, 2) AS ReturnRatePercentage,
    r.Reason,
    COUNT(r.ReturnID) AS ReasonCount
FROM
    BOOKS b
    LEFT JOIN RETURNS r ON b.BookID = r.BookID
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID AND a.AdjustmentType = 'Sale'
GROUP BY
    b.BookID,
    b.Title,
    r.Reason
ORDER BY
    ReturnRatePercentage DESC;

 /*
 Explanation:
 - TotalReturns: Number of return transactions per book.
 - TotalUnitsReturned: Total units returned per book.
 - ReturnRatePercentage: Percentage of units returned out of total units sold.
 - Reason: Common reasons for returns.
 - Helps in identifying problematic products and areas for improvement.

Actionable Steps:
Investigate High Return Rates: Analyze and address issues with books that have high return rates, such as quality problems or mismatched product descriptions.
Improve Product Descriptions: Ensure accurate and detailed product descriptions to set correct customer expectations.
Enhance Quality Control: Implement stricter quality checks for suppliers and incoming stock to reduce defective or damaged products.
Note: Maintaining accurate return data is crucial for the effectiveness of this analysis.
 */

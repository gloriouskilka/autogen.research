-- Monitor Profit Margins
-- Calculate profit margin per book.

SELECT
    b.BookID,
    b.Title,
    b.Price,
    b.Cost,
    (b.Price - b.Cost) AS ProfitPerUnit,
    COALESCE(SUM(-a.QuantityAdjusted), 0) AS TotalUnitsSold,
    COALESCE(SUM((-a.QuantityAdjusted) * (b.Price - b.Cost)), 0) AS TotalProfit
FROM
    BOOKS b
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID AND a.AdjustmentType = 'Sale'
GROUP BY
    b.BookID,
    b.Title,
    b.Price,
    b.Cost
ORDER BY
    TotalProfit DESC;

 /*
 Explanation:
 - ProfitPerUnit: Profit made from each unit sold.
 - TotalUnitsSold: Total number of units sold per book.
 - TotalProfit: Total profit generated from each book.
 - Identifies which books contribute most to profitability.

Actionable Steps:
Adjust Pricing Strategies: Reevaluate pricing for low-margin or loss-making books to improve profitability.
Reduce Costs: Negotiate with suppliers or find cheaper sources to lower the cost of high-margin items further.
Promote High-Margin Products: Focus marketing efforts on products with higher profit margins to maximize returns.

Note: Ensure that the Cost field is accurately maintained for precise profit calculations.
 */

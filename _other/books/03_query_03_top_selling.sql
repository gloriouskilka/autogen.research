-- Identify Top-Selling Books
SELECT
    b.BookID,
    b.Title,
    COALESCE(SUM(CASE WHEN a.AdjustmentType = 'Sale' THEN a.QuantityAdjusted ELSE 0 END), 0) AS TotalUnitsSold,
    COALESCE(SUM(CASE WHEN a.AdjustmentType = 'Sale' THEN a.QuantityAdjusted ELSE 0 END) * b.Price, 0) AS TotalRevenue
FROM
    BOOKS b
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID
        AND a.AdjustmentType = 'Sale'
GROUP BY
    b.BookID,
    b.Title,
    b.Price
ORDER BY
    TotalUnitsSold DESC
LIMIT 10;

/*
Explanation:
- **SUM with CASE Statement**: Explicitly sums only 'Sale' adjustments to calculate `TotalUnitsSold` and `TotalRevenue`.
- **COALESCE**: Handles books with no sales by defaulting sums to zero.
- **LIMIT 10**: Retrieves the top 10 best-selling books.

Actionable Steps:
- **Stock Optimization**: Increase inventory levels for top sellers to meet demand.
- **Marketing Focus**: Allocate more marketing resources towards promoting these high-performing books.
- **Supplier Negotiations**: Leverage sales data to negotiate better terms or discounts with suppliers for high-demand items.
*/

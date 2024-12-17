-- Identify Top-Selling Books
SELECT
    b.BookID,
    b.Title,
    COALESCE(SUM(-a.QuantityAdjusted), 0) AS TotalUnitsSold,
    COALESCE(SUM(-a.QuantityAdjusted) * b.Price, 0) AS TotalRevenue
FROM
    BOOKS b
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID AND a.AdjustmentType = 'Sale'
GROUP BY
    b.BookID,
    b.Title,
    b.Price
ORDER BY
    TotalUnitsSold DESC
LIMIT 10;

 /*
 Explanation:
 - Calculates total units sold and total revenue per book.
 - Orders the results to show the top 10 best-selling books.
 - Useful for identifying high-demand products for potential reordering and promotional focus.

Actionable Steps:
Stock Up Best Sellers: Ensure ample inventory of top-selling books to prevent stockouts.
Promotional Campaigns: Highlight best-sellers in marketing materials to drive further sales.
 */

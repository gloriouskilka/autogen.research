-- Forecast Demand Using Simple Moving Average (Last 6 Months)

SELECT
    b.BookID,
    b.Title,
    COALESCE(SUM(CASE WHEN a.AdjustmentDate >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) AND a.AdjustmentType = 'Sale' THEN -a.QuantityAdjusted ELSE 0 END), 0) AS TotalSalesLast6Months,
    COALESCE(SUM(CASE WHEN a.AdjustmentDate >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) AND a.AdjustmentType = 'Sale' THEN -a.QuantityAdjusted ELSE 0 END), 0) / 6 AS MonthlyAverageSales,
    i.QuantityOnHand,
    (COALESCE(SUM(CASE WHEN a.AdjustmentDate >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) AND a.AdjustmentType = 'Sale' THEN -a.QuantityAdjusted ELSE 0 END), 0) / 6) * 3 AS ForecastNext3Months
FROM
    BOOKS b
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID
    JOIN INVENTORY i ON b.BookID = i.BookID
GROUP BY
    b.BookID,
    b.Title,
    i.QuantityOnHand
ORDER BY
    ForecastNext3Months DESC;

 /*
 Explanation:
 - TotalSalesLast6Months: Total units sold in the last 6 months.
 - MonthlyAverageSales: Average monthly sales based on the last 6 months.
 - ForecastNext3Months: Estimated sales for the next 3 months.
 - Helps in planning inventory purchases based on expected demand.

Actionable Steps:
Adjust Ordering: Use forecasted demand to inform purchase quantities, ensuring you meet future sales without overstocking.
Identify Seasonal Trends: Incorporate seasonal factors into forecasting for more accurate predictions.
Collaborate with Suppliers: Share forecasts with suppliers to secure favorable terms and timely deliveries.
 */

import { isArabicText } from "../arabic.js";

const LABELS = {
  en: { orderId: "Order ID", product: "Product", quantity: "Quantity", unit: "Unit price",
        lineTotal: "Line total", total: "Total", payment: "Payment", cod: "Cash on Delivery", status: "Status" },
  ar: { orderId: "رقم الطلب", product: "المنتج", quantity: "الكمية", unit: "سعر الوحدة",
        lineTotal: "إجمالي السطر", total: "الإجمالي", payment: "الدفع", cod: "الدفع عند الاستلام", status: "الحالة" },
};

// Deliberately excludes customer name/phone/delivery_address -- that PII
// lives only in the persisted order JSON file (src/checkout/demo_order.py),
// never in this on-screen confirmation.
export default function OrderConfirmation({ order, arabic }) {
  if (!order) return null;
  const t = arabic ? LABELS.ar : LABELS.en;

  return (
    <div className="order-confirmation" dir={arabic ? "rtl" : "ltr"}>
      <div className="order-row"><span>{t.orderId}</span><span>{order.order_id}</span></div>
      {order.lines.map((line) => {
        const productName = arabic && line.name_ar ? line.name_ar : line.name_en;
        return (
          <div className="order-line" key={line.id}>
            <div className="order-row">
              <span>{t.product}</span>
              <span dir={isArabicText(productName) ? "rtl" : "ltr"}>{productName}</span>
            </div>
            <div className="order-row"><span>{t.quantity}</span><span>{line.quantity}</span></div>
            <div className="order-row"><span>{t.unit}</span><span>{line.unit_price.toFixed(2)} EGP</span></div>
            <div className="order-row"><span>{t.lineTotal}</span><span>{line.line_total.toFixed(2)} EGP</span></div>
          </div>
        );
      })}
      <div className="order-row"><span>{t.total}</span><span>{order.total_price.toFixed(2)} EGP</span></div>
      <div className="order-row"><span>{t.payment}</span><span>{t.cod}</span></div>
      <div className="order-row"><span>{t.status}</span><span>{order.status}</span></div>
    </div>
  );
}

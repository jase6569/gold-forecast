package uk.jase.goldforecast;

import android.app.PendingIntent;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.widget.RemoteViews;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

/**
 * Home-screen widget for the UK Gold Forecast. Reads data/widget.json, which the
 * evening forecast and the morning Mastermelt check write. Shows tonight's call,
 * Mastermelt's counter prices and tomorrow's likely 9ct range. The last good
 * reading is kept, so a failed refresh never blanks the widget.
 */
public class GoldWidget extends AppWidgetProvider {

    static final String APP_URL = "https://gold-forecast-ten.vercel.app/";
    static final String ACTION_REFRESH = "uk.jase.goldforecast.REFRESH";
    static final int GREY = Color.parseColor("#7D776B");
    static final int AMBER = Color.parseColor("#E3A33B");
    static final int RED = Color.parseColor("#E05D4F");

    static int colourFor(String call) {
        if ("LOCK IN".equals(call)) return Color.parseColor("#F0CB6A");
        if ("WAIT".equals(call)) return Color.parseColor("#3FB27F");
        return Color.parseColor("#CFC8B8");
    }

    @Override
    public void onUpdate(Context ctx, AppWidgetManager mgr, int[] ids) {
        refreshAsync(ctx);
    }

    @Override
    public void onReceive(Context ctx, Intent intent) {
        super.onReceive(ctx, intent);
        if (ACTION_REFRESH.equals(intent.getAction())) refreshAsync(ctx);
    }

    static void requestRefresh(Context ctx) {
        ctx.sendBroadcast(new Intent(ctx, GoldWidget.class).setAction(ACTION_REFRESH));
    }

    private void refreshAsync(Context ctx) {
        final PendingResult pending = goAsync();
        final Context app = ctx.getApplicationContext();
        new Thread(() -> {
            try {
                update(app);
            } catch (Throwable ignored) {
            } finally {
                pending.finish();
            }
        }).start();
    }

    static String get(String url) throws Exception {
        Exception last = null;
        for (int attempt = 0; attempt < 2; attempt++) {
            HttpURLConnection c = null;
            try {
                c = (HttpURLConnection) new URL(url).openConnection();
                c.setConnectTimeout(8000);
                c.setReadTimeout(8000);
                c.setUseCaches(false);
                c.setInstanceFollowRedirects(true);
                c.setRequestProperty("User-Agent", "Mozilla/5.0 (Android) GoldForecastWidget/1.0");
                c.setRequestProperty("Accept", "application/json");
                int code = c.getResponseCode();
                if (code != 200) throw new Exception("HTTP " + code);
                try (BufferedReader r = new BufferedReader(new InputStreamReader(c.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = r.readLine()) != null) sb.append(line);
                    return sb.toString();
                }
            } catch (Exception e) {
                last = e;
                try { Thread.sleep(1200); } catch (InterruptedException ignored) { }
            } finally {
                if (c != null) c.disconnect();
            }
        }
        throw last;
    }

    static String reason(Exception e) {
        String n = e.getClass().getSimpleName();
        if (n.contains("UnknownHost")) return "no internet";
        if (n.contains("Timeout")) return "connection timed out";
        if (n.contains("SSL")) return "secure connection failed";
        if (n.contains("JSON")) return "bad data";
        String m = e.getMessage();
        return (m != null && m.startsWith("HTTP")) ? m : n;
    }

    static void update(Context ctx) {
        AppWidgetManager mgr = AppWidgetManager.getInstance(ctx);
        int[] ids = mgr.getAppWidgetIds(new ComponentName(ctx, GoldWidget.class));
        if (ids.length == 0) return;
        SharedPreferences prefs = ctx.getSharedPreferences("gold", Context.MODE_PRIVATE);
        String time = new SimpleDateFormat("HH:mm", Locale.UK).format(new Date());

        RemoteViews v = new RemoteViews(ctx.getPackageName(), R.layout.widget);
        PendingIntent open = PendingIntent.getActivity(ctx, 0,
                new Intent(Intent.ACTION_VIEW, Uri.parse(APP_URL)), PendingIntent.FLAG_IMMUTABLE);
        PendingIntent refresh = PendingIntent.getBroadcast(ctx, 1,
                new Intent(ctx, GoldWidget.class).setAction(ACTION_REFRESH),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        v.setOnClickPendingIntent(R.id.root, open);
        v.setOnClickPendingIntent(R.id.refresh, refresh);

        RemoteViews busy = new RemoteViews(ctx.getPackageName(), R.layout.widget);
        busy.setTextViewText(R.id.updated, "Refreshing…");
        busy.setTextColor(R.id.updated, GREY);
        mgr.partiallyUpdateAppWidget(ids, busy);

        JSONObject w = null;
        String err = null;
        try {
            w = new JSONObject(get(APP_URL + "data/widget.json?t=" + System.currentTimeMillis()));
            prefs.edit().putString("widget", w.toString()).putString("at", time).apply();
        } catch (Exception e) {
            err = reason(e);
            try {
                String cached = prefs.getString("widget", null);
                if (cached != null) w = new JSONObject(cached);
            } catch (Exception ignored) { }
        }

        if (w == null) {
            v.setTextViewText(R.id.call, "Not loaded yet");
            v.setTextViewText(R.id.updated, "Couldn't connect (" + err + ") at " + time + ". Tap ↻");
            v.setTextColor(R.id.updated, RED);
            mgr.updateAppWidget(ids, v);
            return;
        }

        try {
            v.setTextViewText(R.id.label, w.optString("label", "MASTERMELT 24CT"));
            v.setTextViewText(R.id.price, w.getString("price_line"));
            v.setTextViewText(R.id.call, w.getString("call_line"));
            v.setTextColor(R.id.call, colourFor(w.optString("call")));
            v.setTextViewText(R.id.zone, w.optString("detail_line", ""));

            String upd;
            int col = GREY;
            if (err == null) {
                upd = w.optString("updated_line", "") + " · checked " + time;
            } else {
                upd = "Saved " + prefs.getString("at", "earlier") + " · retry failed (" + err + ")";
                col = AMBER;
            }
            try {
                SimpleDateFormat iso = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm", Locale.UK);
                iso.setTimeZone(TimeZone.getTimeZone("UTC"));
                long hours = (System.currentTimeMillis() - iso.parse(w.getString("generated_utc")).getTime()) / 3600000L;
                if (hours > 80) {  // allows for weekends
                    upd = "⚠ Forecast " + hours + "h old";
                    col = RED;
                }
            } catch (Exception ignored) { }
            v.setTextViewText(R.id.updated, upd);
            v.setTextColor(R.id.updated, col);
        } catch (Exception e) {
            v.setTextViewText(R.id.updated, "Error (" + reason(e) + ") at " + time + ". Tap ↻");
            v.setTextColor(R.id.updated, RED);
        }
        mgr.updateAppWidget(ids, v);
    }
}

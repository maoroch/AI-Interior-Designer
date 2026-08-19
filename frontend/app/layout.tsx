import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Interior Designer",
  description: "AI-платформа для автоматического дизайна интерьера по плану квартиры",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="ru" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}

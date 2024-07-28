import {
  Links,
  Scripts
} from "@remix-run/react";
import type { LinksFunction } from "@remix-run/node";

import rootStyle from "./styles/root.css?url";
import RealTimeIO from "./components/RealTimeIO";

export const links: LinksFunction = () => [
  { rel: "stylesheet", href: rootStyle }
];

export default function App() {
  return (
    <html lang="ja">
      <head>
        <meta charSet="utf-8" />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1"
        />
        <Links />
      </head>
      <body>
        <h1>WoW!</h1>
        <RealTimeIO />
        <Scripts />
      </body>
    </html>
  );
}

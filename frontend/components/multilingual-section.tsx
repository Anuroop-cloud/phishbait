import { Globe } from "lucide-react"

const languages = [
  { name: "Assamese", script: "অসমীয়া", supported: true },
  { name: "Bengali", script: "বাংলা", supported: true },
  { name: "Bodo", script: "बड़ो", supported: true },
  { name: "Dogri", script: "डोगरी", supported: true },
  { name: "English", script: "English", supported: true },
  { name: "Gujarati", script: "ગુજરાતી", supported: true },
  { name: "Hindi", script: "हिन्दी", supported: true },
  { name: "Kannada", script: "ಕನ್ನಡ", supported: true },
  { name: "Kashmiri", script: "कॉशुर", supported: true },
  { name: "Konkani", script: "कोंकणी", supported: true },
  { name: "Maithili", script: "मैथिली", supported: true },
  { name: "Malayalam", script: "മലയാളം", supported: true },
  { name: "Manipuri", script: "মৈতৈলোন", supported: true },
  { name: "Marathi", script: "मराठी", supported: true },
  { name: "Nepali", script: "नेपाली", supported: true },
  { name: "Odia", script: "ଓଡ଼ିଆ", supported: true },
  { name: "Punjabi", script: "ਪੰਜਾਬੀ", supported: true },
  { name: "Sanskrit", script: "संस्कृतम्", supported: true },
  { name: "Santali", script: "ᱥᱟᱱᱛᱟᱲᱤ", supported: true },
  { name: "Sindhi", script: "सिन्धी", supported: true },
  { name: "Tamil", script: "தமிழ்", supported: true },
  { name: "Telugu", script: "తెలుగు", supported: true },
  { name: "Urdu", script: "اردو", supported: true },
]

export function MultilingualSection() {
  return (
    <section id="languages" className="bg-muted py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 lg:px-8">
        <div className="mx-auto mb-14 max-w-2xl text-center">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wider text-primary">
            Multilingual Support
          </p>
          <h2 className="mb-4 text-balance text-3xl font-bold text-foreground md:text-4xl">
            Built for India&apos;s Linguistic Diversity
          </h2>
          <p className="text-pretty text-lg text-muted-foreground">
            SurakshaAI is powered by IndicBERT with broad multilingual coverage across Indian languages.
          </p>
        </div>

        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {languages.map((lang) => (
            <div
              key={lang.name}
              className={`group flex flex-col items-center rounded-2xl border p-6 text-center transition-shadow duration-300 hover:shadow-lg ${
                lang.supported
                  ? "border-border bg-card shadow-sm"
                  : "border-dashed border-border bg-muted"
              }`}
            >
              <div
                className={`mb-3 flex h-12 w-12 items-center justify-center rounded-xl ${
                  lang.supported ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
                }`}
              >
                <Globe className="h-6 w-6" />
              </div>
              <p className="mb-1 text-2xl font-bold text-foreground">{lang.script}</p>
              <p className="text-sm font-medium text-muted-foreground">{lang.name}</p>
              <span className="mt-2 rounded-full bg-accent/10 px-3 py-0.5 text-xs font-medium text-accent">
                Supported
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

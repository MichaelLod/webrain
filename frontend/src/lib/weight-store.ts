const DB_NAME = "webrain-weights";
const STORE_NAME = "shards";
const DB_VERSION = 1;

export interface ShardEntry {
  layerIdx: number;
  component: string; // "gate" | "up" | "down" | "q" | "k" | "v" | "o" | "ln1" | "ln2"
  version: string;
  data: ArrayBuffer;
  storedAt: number;
}

export interface ShardManifest {
  layerIdx: number;
  version: string;
  components: string[];
  totalBytes: number;
}

export class WeightStore {
  private db: IDBDatabase | null = null;

  async open(): Promise<void> {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, {
            keyPath: ["layerIdx", "component", "version"],
          });
          store.createIndex("by_layer_version", ["layerIdx", "version"]);
          store.createIndex("by_version", "version");
        }
      };
      req.onsuccess = () => {
        this.db = req.result;
        resolve();
      };
      req.onerror = () => reject(req.error);
    });
  }

  async storeShard(
    layerIdx: number,
    component: string,
    version: string,
    data: ArrayBuffer,
  ): Promise<void> {
    const db = this.db;
    if (!db) throw new Error("WeightStore not opened");
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put({
        layerIdx,
        component,
        version,
        data,
        storedAt: Date.now(),
      } satisfies ShardEntry);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async getShard(
    layerIdx: number,
    component: string,
    version: string,
  ): Promise<ArrayBuffer | null> {
    const db = this.db;
    if (!db) return null;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get([layerIdx, component, version]);
      req.onsuccess = () => {
        const entry = req.result as ShardEntry | undefined;
        resolve(entry?.data ?? null);
      };
      req.onerror = () => reject(req.error);
    });
  }

  async getManifest(): Promise<ShardManifest[]> {
    const db = this.db;
    if (!db) return [];
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).getAll();
      req.onsuccess = () => {
        const entries = req.result as ShardEntry[];
        const map = new Map<string, ShardManifest>();
        for (const e of entries) {
          const key = `${e.layerIdx}_${e.version}`;
          let m = map.get(key);
          if (!m) {
            m = {
              layerIdx: e.layerIdx,
              version: e.version,
              components: [],
              totalBytes: 0,
            };
            map.set(key, m);
          }
          m.components.push(e.component);
          m.totalBytes += e.data.byteLength;
        }
        resolve(Array.from(map.values()));
      };
      req.onerror = () => reject(req.error);
    });
  }

  async evictVersion(version: string): Promise<void> {
    const db = this.db;
    if (!db) return;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const idx = store.index("by_version");
      const req = idx.openCursor(IDBKeyRange.only(version));
      req.onsuccess = () => {
        const cursor = req.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async clear(): Promise<void> {
    const db = this.db;
    if (!db) return;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
}

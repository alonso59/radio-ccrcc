import axios from 'axios'

export const AUTH_TOKEN_STORAGE_KEY = 'radiology-ui-token'

export type Axis = 'axial' | 'coronal' | 'sagittal'
export type SeriesType = 'nifti' | 'voi'

export interface HealthStatus {
  status: string
}

export interface DatasetSummary {
  dataset_id: string
  path: string
  patient_count: number
  has_nifti: boolean
  has_seg: boolean
  has_voi: boolean
  has_manifest: boolean
}

export interface PatientSummary {
  patient_id: string
  source_patient_id: string | null
  group: string | null
  phases: string[]
  series_count: number
  seg_count: number
  voi_count: number
}

export interface SeriesInfo {
  series_id: string
  patient_id: string
  type: SeriesType
  group: string | null
  phase: string | null
  laterality: string | null
  filename: string
  image_path: string
  mask_path: string | null
  has_seg: boolean
}

export interface VolumeInfo {
  series_id: string
  shape: number[]
  spacing: number[]
  has_mask: boolean
  labels: number[]
}

export interface DatasetViewerSettings {
  last_patient: string | null
  last_series: string | null
  ww: number | null
  wl: number | null
  layers_visible: number[]
  layers_opacity: Record<string, number>
}

export type ViewerSettings = Record<string, DatasetViewerSettings>

export interface SliceQuery {
  ww?: number
  wl?: number
  layers?: number[]
  opacity_1?: number
  opacity_2?: number
  opacity_3?: number
}

type AuthPromptHandler = () => Promise<string | null>

interface RetryableRequestConfig {
  _authRetried?: boolean
}

const api = axios.create({
  baseURL: '/api',
})

let authPromptHandler: AuthPromptHandler | null = null
let pendingPrompt: Promise<string | null> | null = null

api.interceptors.request.use((config) => {
  const token = getStoredAuthToken()
  if (!token) {
    return config
  }

  config.headers = config.headers ?? {}
  config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (!axios.isAxiosError(error) || error.response?.status !== 401 || !error.config) {
      return Promise.reject(error)
    }

    const requestConfig = error.config as typeof error.config & RetryableRequestConfig
    if (requestConfig._authRetried) {
      return Promise.reject(error)
    }

    const token = await requestAuthToken()
    if (!token) {
      return Promise.reject(error)
    }

    setStoredAuthToken(token)
    requestConfig._authRetried = true
    requestConfig.headers = requestConfig.headers ?? {}
    requestConfig.headers.Authorization = `Bearer ${token}`
    return api.request(requestConfig)
  },
)

export function getStoredAuthToken(): string {
  return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) ?? ''
}

export function setStoredAuthToken(token: string): void {
  if (token) {
    window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token)
    return
  }

  window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY)
}

export function registerAuthPromptHandler(handler: AuthPromptHandler | null): void {
  authPromptHandler = handler
}

export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data
    if (typeof detail === 'string') {
      return detail
    }
    if (
      detail &&
      typeof detail === 'object' &&
      'detail' in detail &&
      typeof detail.detail === 'string'
    ) {
      return detail.detail
    }
    if (error.message) {
      return error.message
    }
  }

  if (error instanceof Error && error.message) {
    return error.message
  }

  return 'Unexpected API error'
}

async function requestAuthToken(): Promise<string | null> {
  if (!authPromptHandler) {
    return null
  }
  if (!pendingPrompt) {
    pendingPrompt = authPromptHandler().finally(() => {
      pendingPrompt = null
    })
  }
  return pendingPrompt
}

function buildSliceQuery(query: SliceQuery): string {
  const params = new URLSearchParams()

  if (typeof query.ww === 'number') {
    params.set('ww', String(query.ww))
  }
  if (typeof query.wl === 'number') {
    params.set('wl', String(query.wl))
  }
  if (query.layers && query.layers.length > 0) {
    params.set('layers', query.layers.join(','))
  }

  ;(['opacity_1', 'opacity_2', 'opacity_3'] as const).forEach((key) => {
    const value = query[key]
    if (typeof value === 'number') {
      params.set(key, String(value))
    }
  })

  const queryString = params.toString()
  return queryString ? `?${queryString}` : ''
}

export const apiClient = {
  async getHealth(): Promise<HealthStatus> {
    const response = await api.get<HealthStatus>('/health')
    return response.data
  },

  async listDatasets(): Promise<DatasetSummary[]> {
    const response = await api.get<DatasetSummary[]>('/datasets')
    return response.data
  },

  async listPatients(datasetId: string): Promise<PatientSummary[]> {
    const response = await api.get<PatientSummary[]>(`/datasets/${datasetId}/patients`)
    return response.data
  },

  async listSeries(datasetId: string, patientId: string): Promise<SeriesInfo[]> {
    const response = await api.get<SeriesInfo[]>(
      `/datasets/${datasetId}/patients/${patientId}/series`,
    )
    return response.data
  },

  async loadSeries(
    datasetId: string,
    patientId: string,
    seriesId: string,
  ): Promise<VolumeInfo> {
    const response = await api.post<VolumeInfo>(
      `/datasets/${datasetId}/patients/${patientId}/series/${seriesId}/load`,
    )
    return response.data
  },

  sliceUrl(axis: Axis, index: number, query: SliceQuery = {}): string {
    return `/api/slice/${axis}/${index}${buildSliceQuery(query)}`
  },

  meshUrl(label: number, smooth = true): string {
    const params = new URLSearchParams({ smooth: String(smooth) })
    return `/api/mesh/${label}?${params.toString()}`
  },

  async getSettings(): Promise<ViewerSettings> {
    const response = await api.get<ViewerSettings>('/settings')
    return response.data
  },

  async putSettings(settings: ViewerSettings): Promise<ViewerSettings> {
    const response = await api.put<ViewerSettings>('/settings', settings)
    return response.data
  },
}

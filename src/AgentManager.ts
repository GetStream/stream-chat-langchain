import { Agent } from './Agent';
import type { AgentPlatform } from './types';
import type {
  AgentTool,
  ClientToolDefinition,
} from './LangChainAIAgent';

export interface AgentManagerOptions {
  serverToolsFactory?: () => AgentTool[];
  inactivityThresholdMs?: number;
  cleanupIntervalMs?: number;
  agentIdResolver?: (channelId: string) => string;
}

export interface StartAgentOptions {
  userId: string;
  channelId: string;
  channelType: string;
  platform: AgentPlatform;
  model?: string;
  instructions?: string | string[];
}

export class AgentManager {
  private readonly serverToolsFactory: () => AgentTool[];
  private readonly inactivityThresholdMs: number;
  private readonly cleanupIntervalMs: number;
  private readonly agents = new Map<string, Agent>();
  private readonly pendingAgents = new Set<string>();
  private readonly clientToolRegistry = new Map<string, ClientToolDefinition[]>();
  private readonly cleanupTimer: NodeJS.Timeout;
  private readonly agentIdResolver: (channelId: string) => string;

  constructor(options: AgentManagerOptions = {}) {
    this.serverToolsFactory = options.serverToolsFactory ?? (() => []);
    this.inactivityThresholdMs = options.inactivityThresholdMs ?? 480 * 60 * 1000;
    this.cleanupIntervalMs = options.cleanupIntervalMs ?? 5000;
    this.agentIdResolver = options.agentIdResolver ?? ((channelId) => channelId);
    this.cleanupTimer = setInterval(
      () => void this.cleanupInactiveAgents(),
      this.cleanupIntervalMs,
    );
  }

  get activeAgentCount(): number {
    return this.agents.size;
  }

  getRegisteredClientTools(channelId: string): ClientToolDefinition[] {
    return this.clientToolRegistry.get(channelId) ?? [];
  }

  async startAgent(options: StartAgentOptions): Promise<void> {
    const { userId, channelId } = options;
    let agent = this.agents.get(userId);
    if (!agent) {
      agent = new Agent({
        userId: options.userId,
        channelId: options.channelId,
        channelType: options.channelType,
        platform: options.platform,
        model: options.model,
        instructions: options.instructions,
        serverTools: this.serverToolsFactory(),
      });
      this.agents.set(userId, agent);
    }

    if (this.pendingAgents.has(userId)) {
      console.log(`AI Agent ${userId} already starting`);
      return;
    }

    try {
      this.pendingAgents.add(userId);
      await agent.start();
      const registeredClientTools = this.clientToolRegistry.get(channelId);
      if (registeredClientTools?.length) {
        agent.registerClientTools(registeredClientTools, { replace: true });
      }
    } finally {
      this.pendingAgents.delete(userId);
    }
  }

  async stopAgent(userId: string): Promise<void> {
    const agent = this.agents.get(userId);
    if (!agent) return;
    await agent.stop();
    this.agents.delete(userId);
  }

  registerClientTools(channelId: string, tools: ClientToolDefinition[]): void {
    this.clientToolRegistry.set(channelId, tools);
    const agentId = this.agentIdResolver(channelId);
    const agent = this.agents.get(agentId);
    agent?.registerClientTools(tools, { replace: true });
  }

  dispose(): void {
    clearInterval(this.cleanupTimer);
    this.agents.clear();
    this.pendingAgents.clear();
    this.clientToolRegistry.clear();
  }

  private async cleanupInactiveAgents(): Promise<void> {
    const now = Date.now();
    for (const [userId, agent] of this.agents) {
      if (now - agent.getLastInteraction() > this.inactivityThresholdMs) {
        console.log(`Disposing AI Agent due to inactivity: ${userId}`);
        await agent.stop().catch((error) => {
          console.error(`Failed to stop agent ${userId}`, error);
        });
        this.agents.delete(userId);
      }
    }
  }
}

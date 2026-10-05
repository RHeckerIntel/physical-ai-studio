import { useEffect } from 'react';

import { useRobotCatalogDefinitionQuery } from './robot-catalog.hooks';
import { mapJointToURDFJoint, useLoadModelQuery } from './robot-models-context';
import { SchemaRobotType } from './robot-types';

type JointsState = Array<{
    name: string;
    value: number;
}>;

export const useSynchronizeModelJoints = (joints: JointsState, robotType: SchemaRobotType) => {
    const { data: definition } = useRobotCatalogDefinitionQuery(robotType);
    const jointMap = definition.joint_map;

    const { data: model } = useLoadModelQuery(robotType);

    useEffect(() => {
        if (!model) return;

        joints.forEach((joint) => {
            mapJointToURDFJoint(joint, model, jointMap);
        });
    }, [model, joints, jointMap]);
};

// What drives a follower. 'hold' is the absence of a control, 'policy' a model.
export type FollowerSource = 'hold' | 'teleop' | 'policy';

const RECOVERABLE_ERROR_CODES = new Set(['leader_connection_lost']);

export const isRecoverableRobotControlError = (errorCode: unknown): errorCode is string =>
    typeof errorCode === 'string' && RECOVERABLE_ERROR_CODES.has(errorCode);
